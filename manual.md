# User manual 

* The user must ensure that that they have the data within the correct directory
* Activate the conda environment
* In the terminal, navigate to /path/to/Histomorphological-Phenotype-Learning
* Run `export CUDA_VISIBLE_DEVICES=0; python3 ./run_representationspathology.py --img_size 224 --batch_size 64 --epochs 116 --z_dim 128 --model BarlowTwins_3 --dataset TCGA_PAAD_NO_NORM_TIFF_5x_40pc --check_every 10 --report`
* If the user wishes to use wandb
    * Press 2
    * Enter wandb key
* Else press 3

The results are held within:

/data_model_output/BarlowTwins_3/TCGA_PAAD_NO_NORM_TIFF_5x_40pc/h224_w224_n3_zdim128
