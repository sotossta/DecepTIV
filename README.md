# DecepTIV: Learning to Detect T2V and I2V Videos


Official code of: DecepTIV: Learning to Detect T2V and I2V Videos



## Download the videos from DecepTIV dataset

- Download the dataset from [huggingface](https://huggingface.co/datasets/sotossta/DecepTIV) and place it in `./Dataset/`. The videos from the dataset take around 61GB of memory.
- The `./Dataset/`. folder should contain 11 subfolders corresponding to different T2V or I2V generator and one subfolder for the Real videos.

```
Every subfolder corresponding to a generator or Real videos has the following
structure. For example for HunyuanVideo:

📦HunyuanVideo
 ┣ 📂Firefighter
 ┃ ┣ 📂splits
 ┃ ┣ 📂videos
 ┃ ┗ 📂videos_pert
 ┣ 📂Soldier
 ┃ ┣ 📂splits
 ┃ ┣ 📂videos
 ┃ ┗ 📂videos_pert
 ┗ 📂Weather
 ┃ ┣ 📂splits
 ┃ ┣ 📂videos
 ┃ ┗ 📂videos_pert

The method subfolder contains three more subfolders corresponding to the category
of videos (Firefighter, Soldier and Weather). Then each of these subfolders contains
folders containing the splits, videos and perturbed videos.
 ```

## Extract video frames

To extract a specific number of frames from a given T2V or I2V method (e.g. HunyuanVideo) run the following from the base directory:

```bash
python extract_frames.py \
--base_dir Data /sotossta/DecepTIV/Dataset \
--category all \
--dataset HunyuanVideo \huggingface
--max_frames 50 \
--perturbed 0
```

You can extract the frames from all methods by changing the `dataset` argument to `all`. Extracting the frames from videos will create another subfolder called `/images` (or `/images_pert` for the perturbed videos).

```
The HunyuanVideo directory should now have the following form

📦HunyuanVideo
 ┣ 📂Firefighter
 ┃ ┣ 📂splits
 ┃ ┣ 📂videos
 ┃ ┗ 📂videos_pert
 ┣ 📂Soldier
 ┃ ┣ 📂splits
 ┃ ┣ 📂videos
 ┃ ┗ 📂videos_pert
 ┗ 📂Weather
 ┃ ┣ 📂splits
 ┃ ┣ 📂videos
 ┃ ┗ 📂videos_pert
 ```

## Download detector weights for inference

- All the detector weights can be download from  [google drive](https://drive.google.com/drive/folders/1Zu80bu4cQdLtKs1znEnMcd9VuoauaNmN?usp=sharing)
- Every detector folder has 7 subfolders corresponding to the generation method that the detector has been trained on (all folder means that the detector was trained on all 6 generators)
- Further subfolders indicate what categories of videos the detector was trained on. For example: `/FTCN/all/Firefighter/` corresponds to FTCN trained on videos from all generators and from category firefighter, `/FTCN/all/all/`, corresponds to FTCN trained on videos from all generators and from all categories

## Training and testing detectors

### Environment setup
All of our experiments where done using Python version 3.11, so we recommend
that version.


```bash
# move to classification directory
cd classification

# create and activate environment
python -m venv env
source env/bin/activate

# install requirements
pip install -r requirements.txt

python train.py \
--data_dir_path Data \
--log_interval 200 \
--backbone efficientnet
```

### Training

To train an F3Net detector on all generators and all video categories run:
```bash
python train.py \
        --base_dir /sotossta/DecepTIV \
        --dataset all \
        --category all \
        --detector_config /sotossta/DecepTIV/classification/configs/detectors/f3net.yaml \
        --frames_sampled_real 48 \
        --balanced 1 \
```
Training on different datasets and categories can be achieved by altering the `dataset` and `category` arguments. Also to train different detectors change the path of `detector_config` to the configuration file of another detector.

### Testing

As our experiments involve testing a lot of versions of the same detector, we recommend the use of the provided shell script for testing:

```bash
sh test.sh
```
The shell script contains the following variables that can be altered:

1. `trained_on`: the generator that our given detector was trained on e.g. `all` or `HunyuanVideo`
2. `trained_category`: the category of videos that the detector was trained on e.g. `Firefighter` or `Soldier`
3. `test_dataset`: choose testing videos from a specific generator  `all` or `Gen3`
4. `test_category`: choose category of testing videos_pert
5. `perturbed`: choose to test on perturbed videos or not
6. `weights`: name of the model checkpoint e.g. `model_epoch4_val0.9977.tar`

Note the similar to the training section, different detectors for testing can be chosen by altering `detector_config`.

# Generating fake videos
