# Self-supervised learning in non-small cell lung cancer discovers novel morphological clusters linked to patient outcome and molecular phenotypes
* **[Quiros A.C.<sup>+</sup>, Coudray N.<sup>+</sup>, Yeaton A., Yang X., Chiriboga L., Karimkhan A., Narula N., Pass H., Moreira A.L., Le Quesne J.<sup>\*</sup>, Tsirigos A.<sup>\*</sup>, and Yuan K.<sup>\*</sup> Self-supervised learning in non-small cell lung cancer discovers novel morphological clusters linked to patient outcome and molecular phenotypes. 2022](https://arxiv.org/abs/2205.01931)**

---

**Abstract:**

*Histopathological images provide the definitive source of cancer diagnosis, containing information used by pathologists to identify and subclassify malignant disease, and to guide therapeutic choices. These images contain vast amounts of information, much of which is currently unavailable to human interpretation. Supervised deep learning approaches have been powerful for classification tasks, but they are inherently limited by the cost and quality of annotations. Therefore, we developed Histomorphological Phenotype Learning, an unsupervised methodology, which requires no annotations and operates via the self-discovery of discriminatory image features in small image tiles. Tiles are grouped into morphologically similar clusters which appear to represent recurrent modes of tumor growth emerging under natural selection. These clusters have distinct features which can be identified using orthogonal methods. Applied to lung cancer tissues, we show that they align closely with patient outcomes, with histopathologically recognised tumor types and growth patterns, and with transcriptomic measures of immunophenotype.*

---

## Citation
```
@misc{QuirosCoudray2022,
      title={Self-supervised learning in non-small cell lung cancer discovers novel morphological clusters linked to patient outcome and molecular phenotypes},
      author={Adalberto Claudio Quiros and Nicolas Coudray and Anna Yeaton and Xinyu Yang and Luis Chiriboga and Afreen Karimkhan and Navneet Narula and Harvey Pass and Andre L. Moreira and John Le Quesne and Aristotelis Tsirigos and Ke Yuan},
      year={2022},
      eprint={2205.01931},
      archivePrefix={arXiv},
      primaryClass={cs.CV}        
}
```

## Demo Materials

Slides summarizing methodology and results: 
- [Light-weight version.](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/demos/slides/HPL%20Summary.pdf)
- [High-resolution version.](https://drive.google.com/file/d/1F5ffZqXoNLpT5dgzVLhhCnnspyUe4FPQ/view?usp=sharing)
<p align="center">
  <img src="https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/12589de42685f38630e5b2378c0e6f27e16b3ea3/demos/framework_methodology.jpg" width="500">
</p>


---

## Repository overview

In this repository you will find the following sections: 
1. [WSI tiling process](#WSI-tiling-process): Instructions on how to create H5 files from WSI tiles.
2. [Workspace setup](#Workspace-setup): Details on H5 file content and directory structure.
3. [HPL instructions](README_HPL.md): Step-by-step instructions on how to run the complete methodology.
   1. Self-supervised Barlow Twins training.
   2. Tile vector representations.
   3. Combination of all sets into one H5.
   4. Fold cross validation files.
   5. Include metadata in H5 file.
   6. Leiden clustering.
   7. Removing background tiles.
   8. Logistic regression for lung type WSI classification.
   9. Cox proportional hazards for survival regression.
   10. Correlation between annotations and clusters.
   11. Get tiles and WSI samples for HPCs.
4. [Frequently Asked Questions](#Frequently-Asked-Questions).
5. [TCGA HPL files](#TCGA-HPL-files): HPL output files of paper results.  
6. [Dockers](#Dockers): Docker environments to run HPL steps.
7. [Python Environment](#Python-Environment): Python version and packages.

---

## WSI tiling process
This step divides whole slide images (WSIs) into 224x224 tiles and store them into H5 files. At the end of this step, you should have three H5 files. One per training, validation, and test sets. The training set will be used to train the self-supervised CNN, in our work this corresponded to 60% of TCGA LUAD & LUSC WSIs.

We used the framework provided in [Coudray et al. 'Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning' Nature Medicine, 2018.](https://github.com/ncoudray/DeepPATH/tree/master/DeepPATH_code)
The steps to run the framework are _0.1_, _0.2.a_, and _4_ (end of readme). In our work we used Reinhardt normalization, which can be applied at the same time as the tiling is done through the _'-N'_ option in step _0.1_.

## Workspace setup 
This section specifies requirements on H5 file content and directory structure to run the flow.

In the instructions below we use the following variables and names:
- **dataset_name**: `TCGAFFPE_LUADLUSC_5x_60pc`
- **marker_name**: `he`
- **tile_size**: `224`

### H5 file content specification.
If you are not familiar with H5 files, you can find documentation on the python package [here](https://docs.h5py.org/en/stable/quick.html).

This framework makes the assumption that datasets inside each H5 set will follow the format 'set_labelname'. In addition, all H5 files are required to have the same number of datasets. 
Example:
- File: `hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5`
    - Dataset names: `train_img`, `train_tiles`, `train_slides`, `train_samples`
- File: `hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_validation.h5`
    - Dataset names: `valid_img`, `valid_tiles`, `valid_slides`, `valid_samples`
- File: `hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_test.h5`
    - Dataset names: `test_img`, `test_tiles`, `test_slides`, `test_samples`

### Directory Structure
The code will make the following assumptions with respect to where the datasets, model training outputs, and image representations are stored:
- Datasets: 
    - Dataset folder.
    - Follows the following structure: 
        - datasets/**dataset_name**/**marker_name**/patches_h**tile_size**_w**tile_size**
        - E.g.: `datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224`
    - Train, validation, and test sets:
        - Each dataset will assume that at least there is a training set. 
        - Naming convention: 
            - hdf5_**dataset_name**\_**marker_name**\_**set_name**.h5 
            - E.g.: `datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5`
- Data_model_output: 
    - Output folder for self-supervised trained models.
    - Follows the following structure:
        - data_model_output/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size**
        - E.g.: `data_model_output/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128`
- Results: 
    - Output folder for self-supervised representations results.
    - This folder will contain the representation, clustering data, and logistic/cox regression results.
    - Follows the following structure:
        - results/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size**
        - E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128`
    
## HPL Instructions
The flow consists in the following steps:
1. Self-supervised Barlow Twins training.
2. Tile vector representations.
3. Combination of all sets into one H5.
4. Fold cross validation files.
5. Include metadata in H5 file.
6. Leiden clustering.
7. Removing background tiles.
8. Logistic regression for lung type WSI classification.
9. Cox proportional hazards for survival regression.
10. Correlation between annotations and clusters.
11. Get tiles and WSI samples for HPCs.

You can find the full details [here](README_HPL.md).

---

## Frequently Asked Questions
#### I want to reproduce the paper results.
You can find TCGA files, results, and commands to reproduce them [here](README_replication.md). For any questions regarding the  New York University cohorts, please address reasonable requests to the corresponding authors.

#### I have my own cohort and I want to assign existing clusters to my own WSI.
You can follow steps on how to assign existing clusters in [here](README_additional_cohort.md). These instructions will give you assignation to the same clusters reported in the publication.

#### When I run the Leiden clustering step. I get an \'TypeError: can't pickle weakref objects\' error in some folds.
Based on experience, this error occurs with non-compatible version on numba, umap-learn, and scanpy. The package versions in the python environment should work.
But these alternative package combination works:
```
scanpy==1.7.1 
pynndescent==0.5.0 
numba==0.51.2
```

### If you are having any issue running these scripts, please leave a message on the Issues Github tab.

---

## TCGA HPL files
This section contains the following TCGA files produced by HPL:
1. TCGA LUAD & LUSC WSI tile image datasets.
2. TCGA Self-supervised trained weights.
3. TCGA tile projections.
4. TCGA cluster configurations.
5. TCGA WSI & patient representations. 

For the New York University cohorts, please send reasonable requests to the corresponding authors.

### TCGA LUAD & LUSC WSI tile image datasets
You can find the WSI tile images at:
1. [LUAD & LUSC 60% Background max](https://drive.google.com/drive/folders/18skVh8Vk6zoxG3Se5Vlb7a3EKP2xHXXd?usp=sharing)
2. [LUAD & LUSC 60% Background max 250K subsample](https://drive.google.com/drive/folders/1FuPkMnv6CiDe26doUXfEfQEWShgbmp9P?usp=sharing) for self-supervised model training.

### TCGA Pretrained Models
Self-supervised model weights:
1. [Lung adenocarcinoma (LUAD) and squamous cell carcinoma (LUSC) model](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_LUAD_LUSC_5x/19715020).
2. [PanCancer: BRCA, HNSC, KICH, KIRC, KIRP, LUSC, LUAD](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_PanCancer_5x/19949708).

### TCGA tile vector representations
You can find tile projections for TCGA LUAD and LUSC cohorts at the following locations. These are the projections used in the publication results.
1. [TCGA LUAD & LUSC tile vector representations (background and artifact tiles unfiltered)](https://drive.google.com/file/d/1_mXaTHAF6gb0Y4RgNhJCS2l9mgZoE7gR/view?usp=sharing)
2. [TCGA LUAD & LUSC tile vector representations](https://drive.google.com/file/d/1KEHA0-AhxQsP_lQE06Jc5S8rzBkfKllV/view?usp=sharing)

### TCGA clusters
You can find cluster configurations used in the publication results at:
1. [Background and artifact removal](https://drive.google.com/drive/folders/1K0F0rfKb2I_DJgmxYGl6skeQXWqFAGL4?usp=sharing)
2. [LUAD vs LUSC type classification](https://drive.google.com/drive/folders/1TcwIJuSNGl4GC-rT3jh_5cqML7hGR0Ht?usp=sharing)
3. [LUAD survival](https://drive.google.com/drive/folders/1CaB1UArfvkAUxGkR5hv9eD9CMDqJhIIO?usp=sharing)

### TCGA WSI & patient vector representations
You can find WSI and patient vector representations used in the publication results at:
1. [LUAD vs LUSC type classification](https://drive.google.com/file/d/1K2Fteuv0UrTF856vnJMr4DSyrlqu_vop/view?usp=sharing)
2. [LUAD survival](https://drive.google.com/file/d/13P3bKcmD9C7fvEisArOVOTxf19ko6Xyv/view?usp=sharing)

## Dockers
These are the dockers with the environments to run the steps of HPL. Step **'Leiden clustering'** needs to be run with docker [**2**], all other steps can be run with docker [**1**]:
1. **Self-Supervised models training and projections:**
   - [aclaudioquiros/tf_package:v16](https://hub.docker.com/r/aclaudioquiros/tf_package/tags)
2. **Leiden clustering:**
   - [gcfntnu/scanpy:1.7.0](https://hub.docker.com/r/gcfntnu/scanpy) 
   
## Python Environment
The code uses Python 3.7.12 and the following packages:
```
anndata==0.7.8
autograd==1.3
einops==0.3.0
h5py==3.4.0
lifelines==0.26.3
matplotlib==3.5.1
numba==0.52.0
numpy==1.21.2
opencv-python==4.1.0.25
pandas==1.3.3
Pillow==8.1.0
pycox==0.2.2
scanpy==1.8.1
scikit-bio==0.5.6
scikit-image==0.15.0
scikit-learn==0.24.0
scikit-network==0.24.0
scikit-survival==0.16.0
scipy==1.7.1
seaborn==0.11.2
setuptools-scm==6.3.2
simplejson==3.13.2
sklearn==0.0
sklearn-pandas==2.2.0
statsmodels==0.13.0
tensorboard==1.14.0
tensorflow-gpu==1.14.0
tqdm==4.32.2
umap-learn==0.5.0
wandb==0.12.7
```


