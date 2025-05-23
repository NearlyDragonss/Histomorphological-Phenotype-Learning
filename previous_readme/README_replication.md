# HPL Paper Results - Files and replication.
These instructions will provide the results files for the TCGA data. In the following section we do not include the external cohort from the New York University, for any questions regarding these cohorts please address reasonable requests to the corresponding authors.

## 1. Download TCGA tile images
Download and setup datasets for TCGA LUAD & LUSC and Multi-cancer WSI tile images [here](README.md#TCGA-HPL-files)

After doing this step you should have two directories containing the TCGA LUAD & LUSC and Multi-cancer tile image dataset:
-  `datasets/TCGAFFPE_LUADLUSC_5x_60pc`
-  `datasets/v07_10panCancer_5x`

## 2. Download folder with TCGA tile vector representations and cluster configurations. 
You can directly download the whole `results` folder [here](https://drive.google.com/drive/folders/10Mbo17Nj1jNxK62ZS7HHzgTA_GMzzpBj?usp=sharing). The folder contains the following:
- LUAD & LUSC results:
  1. TCGA tile vector representations (filtered background and artifacts): `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`
  2. `lungtype_nn250`: cluster configurations and results lung type classification. 
  3. `lungtype_nn250_clusterfold4`: cluster configurations and results lung type classification with consistent clusters across classification folds.
  4. `luad_overall_survival_nn250`: cluster configurations and results LUAD survival regression.
  5. `luad_overall_survival_nn250_clusterfold0`: cluster configurations and results LUAD survival regression with consistent clusters across survival folds.
- Multi-cancer results:
  1. TCGA tile vector representations (filtered background and artifacts): `results/BarlowTwins_3/v07_10panCancer_5x/h224_w224_n3_zdim128/hdf5_v07_10panCancer_5x_he_complete_os2_filtered_6cl.h5`
  2. `panC_os2_a6cl_v02_toFilter`: cluster configurations for multi-cancer immune signature correlations and overall survival c-index results.

In our paper, we first run the classification and survival task with different cluster configurations per fold. The purpose of this step is to ensure that defining clusters (HPCs) with different WSI will yield similar results. After this, we locked down a cluster fold by providing the argument `--force_fold`.
This is the difference between `lungtype_nn250` and `lungtype_nn250_clusterfold4`, and also `luad_overall_survival_nn250` and `luad_overall_survival_nn250_clusterfold0`. 

**[Important]** You can find further information on this step in the sections Online Methods - Evaluation and Supplementary Figure 8 from the paper.

Folders with paper results:
1. `lungtype_nn250_clusterfold4`: 
    - Logistic regression performance for lung type classification. For more information on this you can refer to [Step 8 of HPL instructions](README_HPL.md)
    - `alphas_summary_auc_mintiles_100_label1.jpg`: Main figure with performance and statistically significant clusters for different leiden parameter resolutions and alpha penalties.
    - `alpha_10p0_mintiles_100/luad_auc_results_mintiles_100.jpg`: Figure with the alpha penalty used in the paper.
    - `alpha_10p0_mintiles_100/forest_plots/leiden_2p0_stats_all_folds_label1.jpg`: Forest plot with clusters, alpha penalty, and resolution used.
    - `leiden_2p0_fold4`: Directory containing cluster tile samples and WSI with cluster overlays for the paper results.

2. `luad_overall_survival_nn250_clusterfold0`:
   - Cox proportional hazards results for LUAD overall survival analysis. For more information on this you can refer to [Step 9 of HPL instructions](README_HPL.md)
   - `c_index_luad_overall_survival_nn250_clusterfold0_l1_ratio_0.0_mintiles_100.jpg`: Main figure with performance for different leiden parameter resolutions and alpha penalties, l1 ratio of ElasticNet is 0.0.
   - `luad_overall_survival_nn250_clusterfold0_leiden_2.0_alpha_1p0_l1ratio_0p0_mintiles_100`: Directory with results for clusters, alpha penalty, and resolution used.
   - `luad_overall_survival_nn250_clusterfold0_leiden_2.0_alpha_1p0_l1ratio_0p0_mintiles_100/leiden_2p0_stats_all_folds.jpg`: Forest plot for clusters, alpha penalty, and resolution used.
   - `luad_overall_survival_nn250_clusterfold0_leiden_2.0_alpha_1p0_l1ratio_0p0_mintiles_100/KM_leiden_2p0_test.jpg`: Kaplan-Meier plot for TCGA high and low-risk groups, and p-value.
   - `leiden_2p0_fold0`: Directory containing correlation values, correlation figures, cluster tile samples, and WSI with cluster overlays for the paper results.

3. `panC_os2_a6cl_v02_toFilter`:
    - Folder with cluster configuration on our multi-cancer results.
    - [Multi-cancer correlations, overall survival, and cluster tile figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/visualizations_multicancer.ipynb): This notebook contains all code to reproduce the results included in the paper. 

## 3. Running the lung classification and LUAD survival regressions.
The previous folders already contain the results from the paper. Nevertheless, if you wanted to rerun the steps 8 (Logisitic regression for lung classification) and 9 (Cox proportional hazards for survival analysis) from [HPL](README_HPL.md); these are the commands:

Logistic regression for lung classification (different cluster configurations per fold):
```
python3 ./report_representationsleiden_lr.py \
--meta_folder lungtype_nn250_clusterfold4 \
--meta_field luad \
--matching_field slides \
--folds_pickle ./utilities/files/LUADLUSC/lungsubtype_Institutions.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 
```

Logistic regression for lung classification (consistent cluster configuration per fold):
```
python3 ./report_representationsleiden_lr.py \
--meta_folder lungtype_nn250_clusterfold4 \
--meta_field luad \
--matching_field slides \
--folds_pickle ./utilities/files/LUADLUSC/lungsubtype_Institutions.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--force_fold 4
```

Cox proportional hazards for LUAD overall survival analysis (different cluster configurations per fold):
```
python3 ./report_representationsleiden_cox.py \
 --meta_folder luad_overall_survival_nn250 \
 --matching_field samples \
 --event_ind_field os_event_ind \
 --event_data_field os_event_data \
 --folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
 --h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5```
```

Cox proportional hazards for LUAD overall survival analysis (consistent cluster configuration per fold):
```
python3 ./report_representationsleiden_cox.py \
 --meta_folder luad_overall_survival_nn250_clusterfold0 \
 --matching_field samples \
 --event_ind_field os_event_ind \
 --event_data_field os_event_data \
 --folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
 --h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5
 --force_fold 0
```

Cox proportional hazards for LUAD overall survival analysis - Individual resolution and Forest plots:
```
python3 ./report_representationsleiden_cox_individual.py \
--meta_folder luad_overall_survival_nn250 \
--matching_field samples \
--event_ind_field os_event_ind \
--event_data_field os_event_data \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--force_fold 0 \ 
--resolution 2.0 \
--l1_ratio 0.0 \
--alpha 1.0 
```

Cluster tile and WSI samples for lung classification:
- Remember to uncomment lines 58-66 from `report_representationsleiden_samples.py`. This command will replicate the samples in the folder `lungtype_nn250_clusterfold4/leiden_2p0_fold4`.
```
python3 ./report_representationsleiden_samples.py \
--meta_folder lungtype_nn250_clusterfold4 \
--meta_field luad \
--matching_field slides \
--resolution 2.0 \
--fold 4 \
--h5_complete_path results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--dpi 1000 \
--dataset TCGAFFPE_LUADLUSC_5x_60pc
```

Cluster tile and WSI samples for lung classification:
- Remember to uncomment lines 58-66 from `report_representationsleiden_samples.py`. This command will replicate the samples in the folder `lungtype_nn250_clusterfold4/leiden_2p0_fold4`.
```
python3 ./report_representationsleiden_samples.py \
--meta_folder luad_overall_survival_nn250_clusterfold0 \
--meta_field luad \
--matching_field slides \
--resolution 2.0 \
--fold 0 \
--h5_complete_path results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--dpi 1000 \
--dataset TCGAFFPE_LUADLUSC_5x_60pc
```

Further survival analysis and immune signature correlations can be obtained through this [notebook.](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/cluster_correlations.ipynb)

## 4. Multi-cancer immune signature correlations and overall survival c-index
[Multi-cancer correlations, overall survival, and cluster tile figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/visualizations_multicancer.ipynb): This notebook contains all code to reproduce the results included in the paper. 

## 5. Paper figures 
The following notebooks reproduce the paper figures. 

[**Note**] In the paper we also include the external NYU cohorts in some figures. The following notebooks/scripts only use the TCGA cohort data.
For any questions regarding the NYU cohorts please address reasonable requests to the corresponding authors.

1. [Cluster (HPC) correlations figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/cluster_correlations.ipynb).
2. [LUAD vs LUSC figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/paper_figures/visualizations.ipynb).
3. [LUAD Survival figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/paper_figures/visualizations_survival.ipynb).
4. [UMAP and PAGA figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/visualizations_UMAP_PAGA.ipynb).
5. [Multi-cancer correlations and survival figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/visualizations_multicancer.ipynb).

For the [UMAP and PAGA figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/visualizations_UMAP_PAGA.ipynb) notebook, you will need to use a different matplotlib version ` python3 -m pip install matplotlib==3.1.1`.