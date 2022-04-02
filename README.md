# CAN
## Contrast and Aggregation Network for Generalized Zero-shot Learning
### Datasets
Download the dataset (APY/AWA1/CUB/SUN) from the work of [Xian et al. (CVPR2017)](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), and save correspongding data into directory `../data/`. Here, we provide the semantic descriptor for CUB, which is the 1,024-dimensional class embeddings generated from textual descriptions `sent_splits.mat`.  
You also can set the parameter `--dataroot` to place the datasets.
### Environment
Our model is trained on 4× GTX-2080Ti GPUs. The system is window10. The python version is python3.6. You can install package by:  
```
pip install -r requirments.txt
```
### Parameters
All parameters can be found in `parameters.txt`
