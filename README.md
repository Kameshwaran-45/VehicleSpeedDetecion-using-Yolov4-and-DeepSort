
## Badges

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

# VehicleSpeedDetecion-using-Yolov4-and-DeepSort

This repo contains a code to estimate the Vehicle speed using real-time data.

It uses

&bull; Tf save_model which is converted from yolov4 pretrained coco-model to detect objects on each of the video frames

&bull; Deep_SORT tracking algorithm to track those objects over different frames

The tracks of the vehicle pass through the reference line. Vehicle speed will be calculated based on the time taken between the reference lines with a known distance. The calculated speed will be converted based on the frame rate since the time taken will differ depending on the system computation.



## Drive Link to Download Model

Link to download the Model file : https://drive.google.com/drive/folders/1hY7i5e1jWEZqRVpJiDCSfBQ28o3ITchs?usp=sharing

## Quick start

To run this project

1. cloning the repo

```bash
  git clone https://github.com/Kameshwaran-45/VehicleSpeedDetecion-using-Yolov4-and-DeepSort.git
```

2. change the directory

```bash
  cd VehicleSpeedDetecion-using-Yolov4-and-DeepSort/
```

3. Installing the pre-requisites

```bash
  pip install -r requirements.txt
```

4. Move the downloaded model file inside the directory and set the config.json and run main.py

```bash
  python Main.py
```





## Reference

&bull; Github:deep_sort@[Nicolai Wojke nwojke](https://github.com/nwojke/deep_sort)

&bull; Github:tensorflow-yolo-v4@[SoloSynth1](https://github.com/SoloSynth1/tensorflow-yolov4)
