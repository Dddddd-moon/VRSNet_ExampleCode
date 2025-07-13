# Title: VRSNet Example Code

This project is intended to provide a supplementary example and the corresponding demo video of VRSNet to help readers better understand the paper entitled “Vibration Representation and Speed-Joint Network for Machinery Fault Diagnosis under Time-varying Conditions with Sparse Fault Data”

# Diagram

This is the diagram of VRSNet (as shown in the figure below)

<img width="1549" height="672" alt="image" src="https://github.com/user-attachments/assets/2c3b5351-b83a-438b-b0c5-f4a6308bb598" />

# Supplementary demo video

In addition, a demonstration video is included in this project to more intuitively illustrate the effectiveness of VRSNet in few-shot fault diagnosis under time-varying conditions, as shown below.

<img width="1907" height="1080" alt="image" src="https://github.com/user-attachments/assets/745f385c-3df0-4eb9-b9ad-c372b286a334" />

# How to use?

For better reader understanding, the three training phases of VRSNet are implemented separately in this project and can be found in main.py.

# About data processing

The authors declare that the data are private and cannot be shared. Readers may continue to validate VRSNet on their own datasets or on publicly available datasets using the provided example code.

The vibration signals are stored sample by sample in TXT format with comma separators, following the naming convention [Class_RotationalFrequency.txt]. Readers may modify the data processing, sample segmentation, and naming rules as needed for their own use.

# Notice
We have fully provided the training set (included in the train.zip archive) for use in the first-phase self-supervised manifold extraction and the second-phase regression training. Due to GitHub’s file size limitations, only a few sample data are included for the validation and test sets, so readers are expected to construct their own training, validation, and test sets as needed. Nevertheless, the complete three-phase training process—including loss curves, accuracy metrics, and key parameter configurations—has been documented in the History folder located in the root directory. Additionally, the final trained models from the last full training run in this project have been preserved and can also be found in History.
