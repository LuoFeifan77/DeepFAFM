# DeepFAFM

![Deep Frequency Awareness Functional Maps for Robust Shape Matching(TVCG2025)](https://github.com/LuoFeifan77/DeepFAFM/blob/main/figures/teaser.jpg)

## Installation
```bash 
conda create -n fmnet python=3.8 # create new virtual environment\
conda activate fmnet\
pip install -r requirements.txt \  # cuda==11.8!
**or **\
conda env create --name DeepFAFM -f environment.yml
```

## Datasets
For training and testing datasets used in this paper, please refer to the [**ULRSSM**](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link) repository from Dongliang Cao et al. Please follow the instructions there to download the necessary datasets and place them under ../data/:

```Shell
├── data
    ├── FAUST_r
    ├── FAUST_a
    ├── SCAPE_r
    ├── SCAPE_a
    ├── SHREC19_r
    ├── TOPKIDS
    ├── SMAL_r
    ├── DT4D_r
    ├── SHREC20
    ├── SHREC16
    ├── SHREC16_test
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.

## Data precomputation
```python
python preprocess_dataset.py  
```     

## Train
To train the model on a specified dataset.
```python
python train.py --opt options/train/smal.yaml
```
You can visualize the training process in tensorboard.
```bash
tensorboard --logdir experiments/
```


## Test
To test the model on a specified dataset.
```python
python test.py --opt options/test/smal.yaml 
```

# Pretrained models
You can find partial pre-trained models in [checkpoints_ours](https://github.com/LuoFeifan77/DeepFAFM/tree/main/checkpoints_ours) for reproducibility.

# Acknowledgement
The framework implementation is adapted from [Unsupervised Learning of Robust Spectral Shape Matching](https://github.com/dongliangcao/Unsupervised-Learning-of-Robust-Spectral-Shape-Matching/tree/main?tab=readme-ov-file).\
The feature learning network implementation is adapted from [DiffusionNet](https://github.com/nmwsharp/diffusion-net)\
The filter learning network implementation is adapted from [How Powerful are Spectral Graph Neural Networks](https://github.com/GraphPKU/JacobiConv/tree/master).

# Attribution
Please cite our paper when using the code. You can use the following bibtex
```
@article{luo2025deep,
  title={Deep Frequency Awareness Functional Maps for Robust Shape Matching},
  author={Luo, Feifan and Li, Qinsong and Hu, Ling and Wang, Haibo and Xu, Haojun and Liu, Xinru and Liu, Shengjun and Chen, Hongyang},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```




