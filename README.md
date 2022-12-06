# Image SR with Test Time Training

### Setup
A [`environment.yml`](environment.yml) file has been provided to create a Conda environment:

```bash
conda create --name image-sr --file environment.yml
conda activate image-sr
```

Please run the following command to install these dependencies as well:
```bash
pip3 install opencv-python oauthlib hdf5storage ninja lmdb requests timm einops pillow
```

### Training SwinIR
To train the base SwinIR model, please edit the [`train_swinir_sr_classical.json`](options/swinir/train_swinir_sr_classical.json) file with your desired parameters. The file is currently set up for distributed training on the cluster for a 4X SR model. To run the training, please submit a batch job:
```bash
sbatch train.sh
```
If you would like to edit the training job time, machine, or GPU devices, please make the changes in [`train.sh`](train.sh) accordingly.\
**Note:** Please edit the **opt** parameter in [`train.sh`](train.sh) for the command running the Python script for training to the file path in your project directory. If this is not edited, it will not use the options file in your project directory.\
We have provided model weights for 4x SwinIR in the [`superresolution`](superresolution/swinir_sr_classical_patch48_x4/models) folder.

### Training TTT Models
TTT model checkpoints can be trained using [`main_train_test_time.py`](main_train_test_time.py). To use the file, you can run the following command:
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
RESULTS_PATH='...' # Path to text file to save metrics
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
RESULTS_PATH='...' # Path to text file to save metrics
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

       


