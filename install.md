# 1. Create a new conda env with Python 3.12
conda create -n medical-seg python=3.12 -y

# 2. Activate it
conda activate medical-seg

# 3. Install PyTorch first (via conda — handles CUDA automatically)
conda install pytorch==2.6.0 -c pytorch -y        # CPU / MPS
# conda install pytorch==2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y  # NVIDIA GPU

# 4. Install everything else
pip install -r requirements.txt