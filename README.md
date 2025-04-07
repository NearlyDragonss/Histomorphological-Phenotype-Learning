# Readme

These are all of the directories and files I have edited:

/data_manipulation/datasetPyTorch.py

/data_manipulation/utils.py

/demos - not used within this project

/models/activations.py

/models/data_augmentation.py

/models/loss.py

/models/normalization.py

/models/ops.py

/models/optimizer.py

/models/tools.py

/models/utils.py

/models/wandb_utils.py

/models/clustering - not used within this project

/models/evaluation/latent_space.py

/models/mil - not used within this project

/models/networks/attention.py

/models/networks/encoder_contrastive.py

/models/score - not used within this project

/models/selfsupervised/BarlowTwins.py

/models/selfsupervised/BarlowTwinsTraining.py

/models/visualization - not used within this project

/utilities - not used within this project

/previous_readme - not used within this project

The results are held within:
/data_model_output/BarlowTwins_3/TCGA_PAAD_NO_NORM_TIFF_5x_40pc/h224_w224_n3_zdim128


## Build instructions

Here is a link to the dataset: [dataset](https://gla-my.sharepoint.com/:u:/g/personal/2559028d_student_gla_ac_uk/EfSc4f9UCoBMib1bxG1wUHYBHMStPPcLpRhPpyvBqi5woA?e=WGDZce)
Here is a link to the conda-pack environment: [conda-pack](https://gla-my.sharepoint.com/:u:/g/personal/2559028d_student_gla_ac_uk/EQSUQofcEShCivl6MH3RdIQBT-JTQw4i3usjfm3A67_WRA?e=svMWbc)

### Requirements


* Python 3.8
* Conda-pack: hpl-export.tar.gz
* Machine which has at least a 20 series gpu
* Dataset in correct directory


### Build steps

To use conda-pack:
* Download hpl-export.tar.gz
* Create a directory for the environment
* Extract the environment with tar -xvf /path/to/hpl_export.tar.gz -C /path/to/directory
* Run source /path/to/directory/bin/activate
*   This activates the environment
* Run conda-unpack
* Run pip install nvidia-tensorflow==1.15.5+nv22.8
* Run pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
* If you wish to make the environment work like other conda environments
* Run source /path/to/directory/bin/deactivate
*   This deactivates the environment
*   Get the full path to directory
*   Navigate to where conda keeps it’s environments usually something like conda/envs.
*   In envs, create a symlink to the directory you created
*   This looks like ln -s /path/to/hpl-env /path/to/conda/envs
*   You should now be able to activate and deactivate the HPL environment like any other conda environment with
*   Conda activate hpl-env-name
*   Conda deactivate hpl-env-name

Add dataset:
* Unzip hdf5_TCGA_PAAD_NO_NORM_TIFF_5x_40pc_he_.zip
* Go to /path/to/Histomorphological-Phenotype-Learning/datasets/TCGA_PAAD_NO_NORM_TIFF_5x_40pc/he/patches_h224_w224
* Copy coontents from hdf5_TCGA_PAAD_NO_NORM_TIFF_5x_40pc_he_.zip to current directory

### Test steps

* The user must ensure that that they have the data within the correct directory
* Activate the conda environment
* In the terminal, navigate to /path/to/Histomorphological-Phenotype-Learning
* Run `export CUDA_VISIBLE_DEVICES=0; python3 ./run_representationspathology.py --img_size 224 --batch_size 64 --epochs 116 --z_dim 128 --model BarlowTwins_3 --dataset TCGA_PAAD_NO_NORM_TIFF_5x_40pc --check_every 10 --report`
* If the user wishes to use wandb
    * Press 2
    * Enter wandb key
* Else press 3






