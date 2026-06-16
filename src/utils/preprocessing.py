import os
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset, DataLoader

class ACDCDataset(Dataset):
    """
    Each item is a single 2D slice from a patient volume.
    Both ED and ES frames are included, so total samples =
    patients x 2 phases x 18 slices.
    """
    def __init__(self, patient_list, augment=False):
        self.augment = augment
        self.samples = []  # list of (image_slice, mask_slice)

        for patient in patient_list:
            for phase in ("ed", "es"):
                image = patient[f"{phase}_image"]  # (256, 256, 18)
                mask  = patient[f"{phase}_mask"]

                for sl in range(image.shape[2]):
                    img_slice  = image[:, :, sl].astype(np.float32)
                    mask_slice = mask[:, :, sl].astype(np.int64)
                    self.samples.append((img_slice, mask_slice))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, mask = self.samples[idx]

        if self.augment:
            image, mask = self._augment(image, mask)

        # Add channel dim: (1, 256, 256)
        image = torch.tensor(image).unsqueeze(0)
        mask  = torch.tensor(mask)

        return image, mask

    def _augment(self, image, mask):
        import scipy.ndimage as ndi

        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask  = np.flipud(mask).copy()

        # Random rotation +/- 15 degrees
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            image = ndi.rotate(image, angle, reshape=False, order=1)
            mask  = ndi.rotate(mask,  angle, reshape=False, order=0)

        # Random scaling 0.85–1.15
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.85, 1.15)
            h, w  = image.shape
            image = ndi.zoom(image, scale, order=1)
            mask  = ndi.zoom(mask,  scale, order=0)
            # Crop or pad back to original size
            image = _fix_size(image, h, w)
            mask  = _fix_size(mask,  h, w)

        # Random brightness shift (image only)
        if np.random.rand() > 0.5:
            image = image + np.random.uniform(-0.1, 0.1)

        return image, mask

def load_patient(patient_dir):
    """Load ED and ES frames + masks for one patient, reading ED/ES indices from Info.cfg"""
    cfg_path = os.path.join(patient_dir, "Info.cfg")
    info = {}
    with open(cfg_path) as f:
        for line in f:
            key, val = line.strip().split(": ")
            info[key] = val

    pid = os.path.basename(patient_dir)
    ed_idx = int(info["ED"])
    es_idx = int(info["ES"])

    def load(fname):
        return nib.load(os.path.join(patient_dir, fname))

    ed_nii  = load(f"{pid}_frame{ed_idx:02d}.nii.gz")
    ed_gt   = load(f"{pid}_frame{ed_idx:02d}_gt.nii.gz")
    es_nii  = load(f"{pid}_frame{es_idx:02d}.nii.gz")
    es_gt   = load(f"{pid}_frame{es_idx:02d}_gt.nii.gz")

    return {
        "patient_id":  pid,
        "group":       info["Group"],
        "spacing":     ed_nii.header.get_zooms(),   
        "ed_image":    ed_nii.get_fdata(),           
        "ed_mask":     ed_gt.get_fdata().astype(np.uint8),
        "es_image":    es_nii.get_fdata(),
        "es_mask":     es_gt.get_fdata().astype(np.uint8),
    }

def load_split(split_dir):
    """Load all patients from a train or test folder"""
    patients = []
    for pid in sorted(os.listdir(split_dir)):
        patient_path = os.path.join(split_dir, pid)
        if os.path.isdir(patient_path):
            try:
                patients.append(load_patient(patient_path))
                print(f"  Loaded {pid}")
            except Exception as e:
                print(f"  Skipped {pid}: {e}")
    return patients

def resample_volume(image, original_spacing, target_spacing=(1.5, 1.5)):
    """
    Resample a 3D volume (H, W, slices) to a new in-plane spacing.
    Slice thickness is left unchanged (10mm through-plane is too thick for 3D resampling to help).
    """
    zoom_x = original_spacing[0] / target_spacing[0]
    zoom_y = original_spacing[1] / target_spacing[1]

    resampled = ndi.zoom(image, zoom=(zoom_x, zoom_y, 1.0), order=1)  # bilinear for images
    return resampled

def resample_mask(mask, original_spacing, target_spacing=(1.5, 1.5)):
    """Same resampling but nearest-neighbour to preserve integer labels."""
    zoom_x = original_spacing[0] / target_spacing[0]
    zoom_y = original_spacing[1] / target_spacing[1]

    resampled = ndi.zoom(mask, zoom=(zoom_x, zoom_y, 1.0), order=0)  # nearest neighbour
    return resampled

def normalize(image):
    """Z-score normalization per volume (standard for MRI)."""
    mean = image.mean()
    std  = image.std()
    return (image - mean) / (std + 1e-8)

def preprocess_patient(patient, target_spacing=(1.5, 1.5), target_shape=(224, 224, 18)):
    """Resample + normalize + crop/pad — full pipeline for one patient."""
    spacing = patient["spacing"][:2]

    processed = {
        "patient_id": patient["patient_id"],
        "group":      patient["group"],
        "spacing":    target_spacing,
    }

    for phase in ("ed", "es"):
        image = patient[f"{phase}_image"]
        mask  = patient[f"{phase}_mask"]

        image_r = resample_volume(image, spacing, target_spacing)
        mask_r  = resample_mask(mask,  spacing, target_spacing)
        image_n = normalize(image_r)
        image_c = crop_or_pad(image_n, target_shape)
        mask_c  = crop_or_pad(mask_r,  target_shape)

        processed[f"{phase}_image"] = image_c
        processed[f"{phase}_mask"]  = mask_c

    return processed

def crop_or_pad(volume, target_shape=(224, 224, 18)):
    """
    Center crop or pad a 3D volume (H, W, slices) to target shape.
    """
    h, w, slices = volume.shape
    th, tw, ts = target_shape
    result = np.zeros((th, tw, ts), dtype=volume.dtype)

    def get_coords(size, target):
        if size >= target:
            start = (size - target) // 2
            src = (start, start + target)
            dst = (0, target)
        else:
            start = (target - size) // 2
            src = (0, size)
            dst = (start, start + size)
        return src, dst

    (h0, h1), (dh0, dh1) = get_coords(h, th)
    (w0, w1), (dw0, dw1) = get_coords(w, tw)
    (s0, s1), (ds0, ds1) = get_coords(slices, ts)

    result[dh0:dh1, dw0:dw1, ds0:ds1] = volume[h0:h1, w0:w1, s0:s1]
    return result

def _fix_size(arr, target_h, target_w):
    """Crop or pad a 2D array back to (target_h, target_w) after scaling."""
    h, w = arr.shape
    result = np.zeros((target_h, target_w), dtype=arr.dtype)

    def coords(size, target):
        if size >= target:
            start = (size - target) // 2
            return (start, start + target), (0, target)
        else:
            start = (target - size) // 2
            return (0, size), (start, start + size)

    (h0, h1), (dh0, dh1) = coords(h, target_h)
    (w0, w1), (dw0, dw1) = coords(w, target_w)
    result[dh0:dh1, dw0:dw1] = arr[h0:h1, w0:w1]
    return result

if __name__ == "__main__":
    DATASET_ROOT = "data/ACDC/"

    print("Loading training set...")
    train_data = load_split(os.path.join(DATASET_ROOT, "training"))

    print("\nLoading test set...")
    test_data  = load_split(os.path.join(DATASET_ROOT, "testing"))

    print(f"\nDone. {len(train_data)} train patients, {len(test_data)} test patients.")

    train_processed = [preprocess_patient(p) for p in train_data]
    test_processed  = [preprocess_patient(p) for p in test_data]

    shapes = [p["ed_image"].shape for p in train_processed]
    assert len(set(shapes)) == 1, "Not all volumes are the same shape!"
    print(f"All good, shape: {shapes[0]}")

    p = train_processed[2]
    print(f"Patient:        {p['patient_id']}")
    print(f"ED image shape: {p['ed_image'].shape}")
    print(f"ED image mean:  {p['ed_image'].mean():.4f}  (should be ~0)")
    print(f"ED image std:   {p['ed_image'].std():.4f}   (should be ~1)")
    print(f"Mask labels still intact: {np.unique(p['ed_mask'])}")

    train_dataset = ACDCDataset(train_processed, augment=True)
    test_dataset  = ACDCDataset(test_processed,  augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    images, masks = next(iter(train_loader))
    print(f"Image batch: {images.shape}  dtype: {images.dtype}")
    print(f"Mask batch:  {masks.shape}   dtype: {masks.dtype}")
    print(f"Mask labels: {masks.unique()}")
    print(f"Total train slices: {len(train_dataset)}")
    print(f"Total test slices:  {len(test_dataset)}")