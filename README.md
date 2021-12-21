# Self-Supervised Multi-Task Procedure Learning from Instructional Videos

## Overview
This repository contains the implementation of [Self-Supervised Multi-Task Procedure Learning from Instructional Videos](https://khoury.neu.edu/home/eelhami/publications/SelfSupProcLearn-ECCV2020.pdf).

---
## Prerequisites
+ Python 3.x
+ PyTorch 1.x.x
+ sklearn
+ matplotlib
+ skimage
+ scipy

---
## Data Preparation
### CrossTask
1) Please download and extract the annotations and video links of CrossTask to `./data/CrossTask` via https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip

2) To crawl videos in CrossTask from YouTube, please run:
```
python ./download_youtube/download_youtube_CrossTask.py
```

3) To extract and subsample video frame feature, please run:
```
python ./extract_feature/extract_img_frame_CrossTask.py
python ./extract_feature/extract_feature_cats_CrossTask_subsample.py
```

* We use the original training/testing split from CrossTask dataset

### ProceL
1) Please download and extract the annotations and video links of ProceL to `./data/ProceL` via  https://drive.google.com/file/d/1b8PoZlYeNMP3PieJ3_80KkeibaCmmljS/view?usp=sharing

2) To crawl videos in ProceL from YouTube, please run:
```
python ./download_youtube/download_youtube_ProceL.py
```

3) Please run the following preprocessing script to gather annotation file in ProceL:
```
python ./preprocess/rename_gather_mat_file_ProceL.py
``` 

4) To extract and subsample video frame feature, please run:
```
python ./extract_feature/extract_img_frame_ProceL.py
python ./extract_feature/extract_feature_cats_ProceL.py
python ./extract_feature/extract_feature_cats_ProceL_subsample.py
```

* Information about the training split of ProceL is in `./data_partition/ProceL_train_partition.csv`

---
## Training 
### [CrossTask] Task-Specific Setting 
1) To train the task-specific model on CrossTask dataset, please run:
```
chmod +x ./chain_experiment_script/CrossTask/CrossTask_same_cat_ss.sh
./chain_experiment_script/CrossTask/CrossTask_same_cat_ss.sh
```

### [CrossTask] Multi-Task Setting
1) To train the multi-task model on CrossTask dataset, please run:
```
chmod +x ./chain_experiment_script/CrossTask/CrossTask_all_cat_experiments.sh
./chain_experiment_script/CrossTask/CrossTask_all_cat_experiments.sh
```
### [ProceL] Multi-Task Setting
1) To train the multi-task model on ProceL dataset, please run:
```
chmod +x ./chain_experiment_script/ProceL/all_cat_experiments.sh
./chain_experiment_script/ProceL/all_cat_experiments.sh
```

## Pretrained Models
For ease of reproducing the results, we provided the pretrained models for:
Dataset | Setting | Model
--- |:---:|:---:
CrossTask | K=15 (Teacher model) | [download](https://drive.google.com/file/d/1KOSNy2XuxbbtmJBOwvg3Wkn8sJamTSEh/view?usp=sharing)
CrossTask | K=15 (Student model) | [download](https://drive.google.com/file/d/1Z1CLAbs8ymUtTxSNSzgMAe34unMMijo6/view?usp=sharing)
Procel | K=15 (Teacher model) | [download](https://drive.google.com/file/d/1FD4zjin3MmiRSTCiug8kTo2UI6wo87kM/view?usp=sharing)
Procel | K=15 (Student model) | [download](https://drive.google.com/file/d/1uNOHeoqBkpM_geyEmqpp1x3DVuNFgJT8/view?usp=sharing)

---
## Citation
If this code is helpful for your research, we would appreciate if you cite the work:
```
@article{Elhamifar-MultiTaskProcedureLearning:ECCV20,
  author = {E.~Elhamifar and D.~Huynh},
  title = {Self-Supervised Multi-Task Procedure Learning from Instructional Videos},
  journal = {European Conference on Computer Vision},
  year = {2020}}
```
