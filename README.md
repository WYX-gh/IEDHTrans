# IEDHTrans
The code for the paper "IEDHTrans: A Hybrid Network with Interactive Encoders and Differential Hierarchical Transformers for Multi-Phase Breast Cancer Segmentation" published in [Dispalys](https://doi.org/10.1016/j.displa.2025.103193)

# The architecture of IEDHTrans
![image1](https://github.com/WYX-gh/IEDHTrans/blob/main/IEDHTrans/IEDHTrans.jpg) 
![image2](https://github.com/WYX-gh/IEDHTrans/blob/main/IEDHTrans/MPFI.jpg) 
You can obtain the PDF version of the article from [here](https://github.com/WYX-gh/IEDHTrans/blob/main/IEDHTrans/IEDHTrans.pdf)

# Usage
## 1.Installation
Install IEDHTrans as below.
```
git clone https://github.com/WYX-gh/IEDHTrans.git
cd IEDHTrans
pip install -e .
```
After completing the above operations, it is also necessary to install the additional required packages in the [requirements.txt](https://github.com/WYX-gh/IEDHTrans/blob/main/requirements.txt) one by one.  
The requirements.txt contains required packages other than the nnU-Netv2 project.

Note: Our network can also be used in other ways. You can follow the steps below.
Based on your nnU-Netv2 network, place our model file [network_architecture](https://github.com/WYX-gh/IEDHTrans/tree/main/IEDHTrans/nnunetv2/network_architecture) in the nnU-Netv2 folder you are using, and then install the additional required packages in [requirements.txt](https://github.com/WYX-gh/IEDHTrans/blob/main/requirements.txt) to use it.

## 2.Pre-processing
The IEDHTrans use the same pre-processing steps as nnU-Netv2.

Click [here](https://github.com/WYX-gh/IEDHTrans/tree/main/IEDHTrans/documentation) if you need to learn how to use nnU-Netv2.

You can perform data preprocessing through the following command.
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
## 3.Training
Training models is done with the command.
```
nnUNetv2_train DATASET_ID  UNET_CONFIGURATION   FOLD  -tr IEDHTransTrainer
```
UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres, 3d_cascade_lowres). It should be noted that our network only supports 3d input. DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of the 5-fold-cross-validation is trained.
## 4.Validation
Validation is also done with the command.
```
nnUNetv2_predict -d  DATASET_ID  -i   "your_nnUNet_raw_path/imagesTs/"  -o   "your_output_path"   -f FOLD  -c UNET_CONFIGURATION -tr IEDHTransTrainer

```
# Dataset
A total of three datasets were used in our paper,among which a private datasets was breast cancer, and the other two dataset was a publicly available breast tumor dataset. If you wish to download these two publicly available dataset, please refer to the [paper](https://arxiv.org/abs/2406.13844v1) , the [GitHub](https://github.com/LidiaGarrucho/MAMA-MIA) and the [paper](https://ieeexplore.ieee.org/document/10614219), the [GitHub](https://github.com/ZhouL-lab/PLHN).
Additionally, We have made public an available internal clinical breast cancer [dataset](https://drive.google.com/file/d/1KwgF698k9VrfwyT2vf0wbmJa9m6rCKeF/view?usp=drive_link).

```
data：
├── image
│   ├── qfs_01_0000.nii.gz
│   ├── qfs_01_0001.nii.gz
│   ├── qfs_02_0000.nii.gz
│   ├── qfs_02_0001.nii.gz
│   ├── qfs_03_0000.nii.gz
│   ├── qfs_03_0001.nii.gz
│   ├── ···
│   ├── ···
│   ├── qfs_31_0000.nii.gz
│   └── qfs_31_0001.nii.gz
└── label
    ├── qfs_01.nii.gz
    ├── qfs_02.nii.gz
    ├── qfs_03.nii.gz
    ├──···
    └── qfs_31.nii.gz
```
Among them, _0000 is the pre-contrast period and _0001 is the first post-contrast period.

# Results
Experimental results on the MAMA-MIA dataset: 
![image3](https://github.com/WYX-gh/IEDHTrans/blob/main/IEDHTrans/Table1_MAMA-MIA%20datasets.png) 
Experimental results on the PLHN dataset: 
![image4](https://github.com/WYX-gh/IEDHTrans/blob/main/IEDHTrans/Table2_PLHN%20datasets.png) 
Experimental results on the our internal dataset: 
![image5](https://github.com/WYX-gh/IEDHTrans/blob/main/IEDHTrans/Table3_internal%20datasets.png)


# Baseline Models
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)  
[nnFormer](https://github.com/282857341/nnFormer)  
[PA-Net](https://github.com/Houjunfeng203934/PA-Net)  
[PLHN](https://github.com/ZhouL-lab/PLHN)  
[PHTrans](https://github.com/lseventeen/PHTrans)  
[STU-Net](https://github.com/uni-medical/STU-Net)

# Contact
Should you have any question, please create an issue on this repository or contact us for more information.  
Yuexin Wang (yxwang5212@outlook.com)


# Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.

# Citation
If you use IEDHTrans for your research or find this dataset useful for your research, please cite our papers:
```
@article{WANG2026103193,
title = {IEDHTrans: A hybrid network with interactive encoders and differential hierarchical transformers for multi-phase breast cancer segmentation},
journal = {Displays},
volume = {91},
pages = {103193},
year = {2026},
issn = {0141-9382},
doi = {https://doi.org/10.1016/j.displa.2025.103193},
author = {Yuexin Wang and Gesheng Song and Jian Zhang and Fangqing Wang and Haixing Cheng and Yudan Zhao and Peng Zhou and Xu Qiao and Wei Chen}
}
```
