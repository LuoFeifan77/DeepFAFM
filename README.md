# DeepFAFM
Implementations for Deep Frequency Awareness Functional Maps for Robust Shape Matching(TVCG2025).


# Installation
conda create -n fmnet python=3.8 # create new virtual environment\
conda activate fmnet\
pip install -r requirements.txt \
or 
conda env create --name DeepFAFM -f environment.yml

# Datasets
For training and testing datasets used in this paper, please refer to the ULRSSM repository from Dongliang Cao et al. Please follow the instructions there to download the necessary datasets and place them under ../data/:

├── data
    ├── FAUST_r
    ├── FAUST_a
    ├── SCAPE_r
    ├── SCAPE_a
    ├── SHREC19_r
    ├── TOPKIDS
    ├── SMAL_r
    ├── DT4D_r


# Data precomputation
python preprocess_dataset.py     

# Train
python train.py --opt options/train/smal.yaml

# Test
python train.py --opt options/test/smal.yaml

# Pretrained models
You can find partial pre-trained models in checkpoints_ours for reproducibility.

# Acknowledgement
The framework implementation is adapted from [Unsupervised Learning of Robust Spectral Shape Matching](https://github.com/dongliangcao/Unsupervised-Learning-of-Robust-Spectral-Shape-Matching/tree/main?tab=readme-ov-file).


The implementation of [DiffusionNet](https://github.com/nmwsharp/diffusion-net) is based on the official implementation.




