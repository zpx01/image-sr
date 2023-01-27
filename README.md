# Image SR with Test Time Training

### Setup
A [`environment.yml`](environment.yml) file has been provided to create a Conda environment:

```bash
conda env create -f environment.yml
conda activate image-sr
```

Please run the following command to install these dependencies as well:
```bash
pip3 install opencv-python oauthlib hdf5storage ninja lmdb requests timm einops pillow
```
\
For the datasets, please check the [`trainsets`](trainsets) and  [`testsets`](testsets) folder. The training data has been obtained from the
[`DIV2K`](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset and the testsets include [`Set5`](testsets/Set5), [`Set14`](testsets/Set14), and [`BSD100`](testsets/BSD_100/). 

### Training SwinIR
To train the base SwinIR model, please edit the [`train_swinir_sr_classical.json`](options/swinir/train_swinir_sr_classical.json) file with your desired parameters. The file is currently set up for distributed training on the cluster for a 4X SR model. To run the training, please submit a batch job:
```bash
sbatch train.sh
```
If you would like to edit the training job time, machine, or GPU devices, please make the changes in [`train.sh`](train.sh) accordingly.\
**Note:** Please edit the **opt** parameter in [`train.sh`](train.sh) for the command running the Python script for training to the file path in your project directory. If this is not edited, it will not use the options file in your project directory.\
We have provided model weights for 4x SwinIR in the [`superresolution`](superresolution/swinir_sr_classical_patch48_x4/models) folder.

### Training TTT Models
TTT model checkpoints can be trained using [`main_test_time.py`](main_test_time.py). To use the file, you can run the following command:
```bash
MODEL_PATH='...'
OPTIMIZER_PATH='...'
TESTSET_DIR='...'
OUTPUT_DIR='...'
python3  -m torch.distributed.launch --nproc_per_node=8 main_test_time.py \
        --model_path ${MODEL_PATH} \
        --opt_path ${OPTIMIZER_PATH} \
        --scale 4 \
        --num_images 10 \
        --epochs 5 \
        --test_dir ${TESTSET_DIR} \
        --output_dir ${OUTPUT_DIR}
```
We will use these TTT models to generate new TTT inferences in the next section. 

### Model Inference
To test the models you have trained, you may run the following commands:
For SwinIR Inference:\
```bash
TASK='classical_sr'
TYPE='swinir'
MODEL_PATH='...' # SwinIR pretrained model path
TEST_FOLDER_LQ='...' # Low quality images for testing
TEST_FOLDER_GT='...' # High quality ground truth images
RESULTS_PATH='...' # Path to folder to save results in
IMG_ID='...' # Unique identifier to use for saved image file paths
python3 main_test_swinir.py \
        --task ${TASK} \
        --type ${TYPE} \
        --scale 4 \
        --training_patch_size 48 \
        --model_path ${MODEL_PATH} \
        --folder_lq ${TEST_FOLDER_LQ} \
        --folder_gt ${TEST_FOLDER_GT} \
        --results_path ${RESULTS_PATH} \
        --img_identifier ${IMG_ID}
```
\
For TTT inference:
```bash
TASK='classical_sr'
TYPE='ttt'
MODELS_DIR='...' # Directory containing all TTT checkpoints to test
TEST_FOLDER_LQ='...' # Low quality images for testing
TEST_FOLDER_GT='...' # High quality ground truth images
RESULTS_PATH='...' # Path to folder to save results in
IMG_ID='...' # Unique identifier to use for saved image file paths
python3 main_test_swinir.py \
        --task ${TASK} \
        --type ${TYPE} \
        --scale 4 \
        --training_patch_size 48 \
        --models_dir ${MODELS_DIR} \
        --folder_lq ${TEST_FOLDER_LQ} \
        --folder_gt ${TEST_FOLDER_GT} \
        --results_path ${RESULTS_PATH} \
        --img_identifier ${IMG_ID}
```
\
Your results will be saved in `results/swinir_{TASK}_x{SCALE}_{IMG_ID}`.

### Merging Model Results
In order to use your results from TTT, we've create the MergeResults class in [`merge_results.py`](merge_results.py). This class allows users to take inference outputs from SwinIR and TTT and take the best pixels from each to create a new merged image. You can run the script with the following command:\
```bash
PRETRAINED_DIR='...' # Directory with pretrained inference images
TTT_DIR='...' # Directory with TTT inference images
GT_DIR='...' # Directory with ground truth images
MERGED_DIR='...' # Directory to store merged results in
RESULTS_LOG='...' # File path for text file to store result metrics
python3 merge_results.py \
        --pretrained_dir ${PRETRAINED_DIR} \
        --ttt_dir ${TTT_DIR} \
        --gt_dir ${GT_DIR} \
        --merged_dir ${MERGED_DIR} \
        --results_log ${RESULTS_LOG}
```
To make sure the script will run as intended, please make sure that the corresponding pretrained, ttt, and gt images are in the correct order in their directories (if the first pretrained image in the pretrained directory is img1, then the first image in the ttt and gt directories must also be img1). 

### Visualizations
You may run the [`overlay_imgs.py`](overlay_imgs.py) script to generate visualizations of the results of your new models. These visualizations include an image with an overlaid mask where red pixels represent that the pretrained SwinIR image had a lower L2 loss and green pixels represent that the TTT image had a lower L2 loss with the ground truth. This bitmask helps identify the regions in which the SwinIR and TTT images perform better than the other. You may set the threshold for the confidence interval manually, or use the default value set at `0.001`.

```bash
THRESH=0.001 # Threshold for L2 loss confidence interval (Set to 0.001 by default)
PRETRAINED='...' # Directory with pretrained inference images
TTT='...' # Directory with TTT inference images
GT='...' # Directory with ground truth images
LR='...' # Directory with low resolution images
RESULTS_DIR='...' # File path to store resulting images

python3 overlay_imgs.py \
	--thresh ${THRESH} \
	--pretrained ${PRETRAINED} \
	--ttt ${TTT} \
	--gt ${GT} \
	--lr ${LR} \
	--results_dir ${RESULTS_DIR}
```

       


