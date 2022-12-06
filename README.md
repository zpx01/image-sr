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




