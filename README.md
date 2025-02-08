# Spener


This is the official code repository of our work **"Unsupervised Self-prior Embedding Implicit Neural Respresentation for Iterative Sparse-View CT Reconstruction"** accepted by AAAI 2025.

## Overview
![Overview of proposed Spener](Fig/pipeline.png)
*Fig. 1. Overview of proposed Spener.* 

## Results
![Reconstruction of ](Fig/results1.png)
*Fig. 2. Qualitative results of CT image reconstructed by the compared methods under two dose settings, with both results reconstructed from AAPM dataset with 90 views.* 

## File Tree
```bash
Spener
│   dataset.py                  # dataloader
│   netarch.py                  # network architecture
│   README.md                   # readme file
│   train.py                    # train file
│   utils.py                    # tools
│
├── config
│       demo_config.json        # demo configuration file
│
├── demo_data
│       demo_AAPM_slice.nii.gz  # demo data 
│
├── Fig
│       pipeline.png     
│       results1.png     
│
└── output
    └── SV_90_I0_1e+06
            final_recon.nii.gz   # Spener reconstruction
            gt_img.nii.gz        # GT image
            sv_fbp.nii.gz        # reconstruction from sparse-view
```


## Main Requirements
To run this project, you will need the following packages:
- PyTorch
- tinycudann
- torch-radon
- bm3d
- SimpleITK
- tqdm
- NumPy 
- other dependencies
  
the repo of torch-radon is at https://github.com/carterbox/torch-radon.
  
## Training and Checkpoints
To train Spener from scratch, navigate to the project directory and run the following command in your terminal:


```bash
python train.py
```

The code will train the Spener to reconstruct a demo slice `./demo_data/demo_AAPM_slice.nii.gz` from the AAPM dataset under the following conditions:
- 90 projection views 
- Normal dose settings ($I_0=10^6$)  

The training config file is `./config/demo_config.json`.  The code will save the corresponding model weights and reconstruction results in each iteration in `./output`. 


## Others

NIFTI files (`.nii`) can be viewed using the ITK-SNAP software, available for free download at: http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4


## Citation
