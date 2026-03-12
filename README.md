# Automated-Detection-Hyalessa-maculaticollis
Automated Detection of Hyalessa maculaticollis from Heterogeneous Citizen-Science Acoustic Recordings

# Citizen-Science[earth-love-explorers]

[Citizen-Science Project Homepage](https://jisatam.dongascience.com/main?utm_source=naver_keyword&utm_medium=cpc&utm_content=earth_project_pc&utm_term=%EC%A7%80%EA%B5%AC%EC%82%AC%EB%9E%91%ED%83%90%EC%82%AC%EB%8C%80&n_media=27758&n_query=%EC%A7%80%EA%B5%AC%EC%82%AC%EB%9E%91%ED%83%90%EC%82%AC%EB%8C%80&n_rank=1&n_ad_group=grp-a001-01-000000047358480&n_ad=nad-a001-01-000000347451557&n_keyword_id=nkw-a001-01-000006781569813&n_keyword=%EC%A7%80%EA%B5%AC%EC%82%AC%EB%9E%91%ED%83%90%EC%82%AC%EB%8C%80&n_campaign_type=1&n_ad_group_type=1&n_match=1)

[Ios APP](https://apps.apple.com/kr/app/%EC%A7%80%EA%B5%AC%EC%82%AC%EB%9E%91%ED%83%90%EC%82%AC%EB%8C%80/id6670779182)

[Android APP](https://play.google.com/store/search?q=%EC%A7%80%EA%B5%AC%EC%82%AC%EB%9E%91%ED%83%90%EC%82%AC%EB%8C%80&c=apps&hl=ko)


## Project Pipeline
*  **01. File Convert**: Unified various raw input files into a standardized format for seamless downstream processing
*  **02. Drawing linear Spectrogram**: Generated linear spectrograms from the standardized audio data, transforming time-frequency information into 2D visual inputs suitable for CNN model training
*  **03. Spectrogram Segmentation**: Segmented the spectrograms based on the acoustic characteristics of the target species to generate focused data for the model
*  **04. Resizing**: Resized the segmented spectrogram data to meet the input requirements of the ResNet model
*  **05. Labeling**: Assigned labels to indicate the presence of the target species, establishing the ground truth for model training
*  **06. Data Splitting**: Divided the dataset into training and evaluation sets to ensure an unbiased performance measurement
*  **07. Model Training**: Conducted the training process using the ResNet architecture on the preprocessed spectrogram data
*  **08. Model Evaluation**: Evaluated the trained model's performance to measure its accuracy and effectiveness in detecting the target species

> The source code for each stage of the pipeline is available in the code directory. Please note that certain scripts have been withheld from this public repository to protect the personal information of the citizen scientists involved in data collection. If you require access to the full source code for research or verification purposes, please contact me.
  
## Environment
The learning environment for this project is as follows:

### Hardware
*   **GPU**: NVIDIA GeForce RTX 5070 (12GB) x 1
*   **CPU**: Intel(R) Core(TM) Ultra 7 265K @ 3.90 GHz
*   **RAM**: 32GB

### Software
*   **OS**: Window 11
*   **Python version**: Python 3.11.14
*   **CUDA version**: 12.8
*   **cuDNN version**: 9.10.2

### Dependencies
**Deep Learning Framework**
* `torch`: 2.11.0.dev20251217+cu128
* `torchaudio`: 2.10.0.dev20251217+cu128
* `torchvision`: 0.25.0.dev20251217+cu128

**Data Processing & Math**
* `numpy`: 2.2.6
* `pandas`: 2.3.3
* `scipy`: 1.16.3
* `scikit-learn`: 1.8.0

**Audio & Signal Processing**
* `librosa`: 0.11.0
* `soundfile`: 0.13.1

**Image & Visualization**
* `matplotlib`: 3.10.8
* `seaborn`: 0.13.2
* `opencv-python`: 4.12.0.88
* `ImageIO`: 2.37.2
* `grad-cam`: 1.5.5

**Utilities**
* `tqdm`: 4.67.1
* `imageio-ffmpeg`: 0.6.0 


## Contact

For access to the full source code or inquiries regarding the citizen science dataset, please reach out via email:

| Name | Email |
| :--- | :--- |
| Your Name | [e.jaeyun82@gmail.com](mailto:e.jaeyun82@gmail.com) |
