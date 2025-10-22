# Intro

Our codes are based on https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark. and https://github.com/chihina/GAFL-CVPR2024.
I deeply appreciate their efforts.

This is the official repository for the following paper:

Chihiro Nakatani, Hiroaki Kawashima, Norimichi Ukita  
Human-in-the-loop Adaptation in Group Activity Feature Learning for Team Sports Video Retrieval, (under review).

## Environment
python 3.10.2  
ROIAlign (https://github.com/longcw/RoIAlign.pytorch)

And you can use requirements.txt
```
pip install -r requirements.txt
```

# Data preparation
## 1. Download dataset
You can download daatset from the following url.  
These dataset are required to place in data/ in the repository as follows:

* Volleyball dataset (data/volleyball/videos)  
https://github.com/mostafa-saad/deep-activity-rec

* NBA dataset (data/basketball/videos)  
https://ruiyan1995.github.io/SAM.html

* Collective Activity dataset (data/collective)  
https://cvgl.stanford.edu/projects/collective/collectiveActivity.html


## 2. Training
* You can change parameters of the model by editing the files located in scripts (e.g., scripts/run_multiple_volleyball.bash).

### 2.1 Volleyball dataset

* Ours
```
bash scripts/run_multiple_volleyball.bash
```

### 2.2 NBA dataset

* Ours
```
bash scripts/run_multiple_basketball.bash
```

### 2.3 Collective Activity dataset

* Ours
```
bash scripts/run_multiple_volleyball.bash
```