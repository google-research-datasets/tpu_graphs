!#/bin/bash
# running on RTX 4090 instance of vast.ai

# download & init minoconda3
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash

source ~/.bashrc

# create tpugraph environment
conda create -n tpugraphs python=3.10 -y
conda activate tpugraphs

conda install -c conda-forge cudatoolkit=11.8.0 -y
pip install nvidia-cudnn-cu11==8.6.0.163


conda install -c conda-forge tensorflow -y
conda install -c conda-forge tqdm -y

pip install  tensorflow_gnn --pre
pip install tensorflow-ranking
conda clean --all -y

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

git clone https://github.com/google-research-datasets/tpu_graphs.git

# download train&valid&test dataset
cd tpu_graphs
python3 echo_download_commands.py | bash

# upgrade system environment
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update -y
apt-get install --only-upgrade libstdc++6 -y

# apt-get install libcublas-12-0 -y
apt-get install vim -y


pip install kaggle
mkdir ~/.kaggle
echo "{\"username\":\"yyq263\",\"key\":\"5ab15f2cde2b024a1a11ac99ff218cd9\"}" > ~/.kaggle/kaggle.json

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib:$CONDA_PREFIX/lib/
