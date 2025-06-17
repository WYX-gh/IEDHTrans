# IEDHTrans

# Usage
## 1.Installation
Install PEHTrans as below
```
git clone https://github.com/WYX-gh/IEDHTrans.git
pip install -e .
pip install -r requirements.txt
```
The requirements.txt contains third-party libraries other than the nnunet project

Note: Our network can also be used in other ways. You can follow the steps below.
Based on the nnU-Netv2 network, place our model file [network_architecture](https://github.com/WYX-gh/IEDHTrans/tree/main/IEDHTrans/nnunetv2/network_architecture) in the nnunet folder you are using, and then install the additional third-party libraries in requirement.txt to use it.

## 2.Pre-processing
All compared methods use the same pre-processing steps as nnUNet.

Click [here]((https://github.com/WYX-gh/IEDHTrans/tree/main/IEDHTrans/documentation)) if you need to learn how to use nnUNet.

You can perform data preprocessing through the following command
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
## 3.Training
Training models is done with the command.
```
nnUNetv2_train DATASET_ID  UNET_CONFIGURATION   FOLD  -tr IEDHTransTrainer
```
UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres, 3d_cascade_lowres).It should be noted that our network only supports 3d input. DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of the 5-fold-cross-validation is trained.
## 4.Validation
Validation is also done with the command.
```
nnUNetv2_predict -d  DATASET_ID  -i   "your_nnUNet_raw_path/imagesTs/"  -o   "your_output_path"   -f FOLD  -c UNET_CONFIGURATION -tr IEDHTransTrainer

```
# Dataset
A total of two datasets were used in our paper,among which a private datasets was breast cancer, and the other dataset was a publicly available breast tumor dataset.If you wish to download this publicly available dataset, please refer to the relevant [paper](https://arxiv.org/abs/2406.13844v1) and the [GitHub](https://github.com/LidiaGarrucho/MAMA-MIA)

Additionally, We have made public an available internal clinical breast cancer [dataset]

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
Among them, _0000 is the pre-contrast period and _0001 is the first post-contrast period



# Baseline Models
[nnUNet](https://github.com/MIC-DKFZ/nnUNet)  
[nnFormer](https://github.com/282857341/nnFormer)  
[PA-Net](https://github.com/Houjunfeng203934/PA-Net)  
[PLHN](https://github.com/ZhouL-lab/PLHN)  
[PHTrans](https://github.com/lseventeen/PHTrans)
[STU-Net](https://github.com/uni-medical/STU-Net)

# Contact
For any query, please contact us for more information.  
Yuexin Wang (ghw231784@gmail.com)


# Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.
