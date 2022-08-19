## Tutorial Handbook
### Input data format

Bulk2Space requires five formatted data as input:
1. Bulk-seq Normalized Data: a `.csv` file with genes as rows and one sample as column

|<img width=140/> <img width=140/>|<img width=139/>Sample<img width=138/>| 
| :-----: | :-----: | 
| Gene1 | 5.22 |
| Gene2 | 3.67 |
| ... | ... |
| GeneN | 15.76 |

****  
2. Single Cell RNA-seq Normalized Data: a `.csv` file with genes as rows and cells as columns
  
|<img width=40/> <img width=40/>|<img width=25/>Cell1<img width=25/>|<img width=25/>Cell2<img width=25/>|<img width=25/>Cell3<img width=25/>|<img width=35/>...<img width=34/>|<img width=25/>CellN<img width=25/>| 
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| Gene1 | 1.05 | 2.31 | 1.72 | ... | 0 |
| Gene2 | 4.71 | 1.07 | 0 | ... | 4.22 |
| ... | ... | ... | ... | ... | ... |
| GeneN | 0.55 | 0 | 1.48 | ... | 0 |

****
3. Single Cell RNA-seq Annotation Data: a `.csv` file with cell ID and celltype annotation columns. 
   * The column containing cell ID should be named `Cell` 
   * the column containing the labels should be named `Cell_type` 

|<img width=90/> <img width=90/>|<img width=80/>Cell<img width=81/>|<img width=75/>Cell_type<img width=75/>|
| :-----: | :-----: | :-----: |
| Cell1 | Cell1 | T cell |
| Cell2 | Cell2 | B cell |
| ... | ... | ... |
| CellN | CellN | Monocyte |

****
4. Spatial Transcriptomics Normalized Data: a `.csv` file with genes as rows and cells (or spots) as columns 

|<img width=45/> <img width=45/>|<img width=20/>Cell1 / Spot1<img width=10/>|<img width=20/>Cell2 / Spot2<img width=20/>|<img width=33/>...<img width=32/>|<img width=15/>CellN / SpotN<img width=15/>| 
| :-----: | :-----: | :-----: | :-----: | :-----: |
| Gene1 | 3.22 | 4.71 | ... | 1.01 |
| Gene2 | 0 | 2.17 | ... | 2.20 |
| ... | ... | ... | ... | ... |
| GeneN | 0 | 0.11 | ... | 1.61 |

****
5. Spatial Transcriptomics Coordinates Data: a `.csv` with cell/spot ID and coordinates columns. 
   * The column containing the coordinates should be named `xcoord` and `ycoord`
   * For spot-based data, the column containing spot ID should be named `Spot`
   * For image-based data, the column containing cell ID should be named `Cell`
  
|<img width=50/> <img width=50/>|<img width=44/>Spot (or Cell) <img width=44/>|<img width=44/>xcoord<img width=45/>|<img width=45/>ycoord<img width=45/>|
| :-----: | :-----: | :-----: | :-----: |
| Cell_1 / Spot_1 | Cell_1 / Spot_1 | 1.2 | 5.2 |
| Cell_2 / Spot_2 | Cell_1 / Spot_1 |5.4 | 4.3 |
| ... | ... | ... | ... |
| Cell_n / Spot_n | Cell_1 / Spot_1 | 11.3 | 6.3 |

****

### Parameter description
* Decompose bulk transcriptomics data into single-cell transcriptomics data:
```python
from bulk2space import Bulk2Space
model = Bulk2Space()

# Decompose bulk transcriptomics data into single-cell transcriptomics data
generate_sc_meta, generate_sc_data = model.train_vae_and_generate(
    input_bulk_path,
    input_sc_data_path,
    input_sc_meta_path,
    input_st_data_path,
    input_st_meta_path,
    ratio_num=1,
    top_marker_num=500,
    gpu=0,
    batch_size=512,
    learning_rate=1e-4,
    hidden_size=256,
    epoch_num=5000,
    vae_save_dir='save_model',
    vae_save_name='vae',
    generate_save_dir='output',
    generate_save_name='output')
```

| Parameter | Description | Default Value |
| --- | --- | --- |
| input_bulk_path | Path to bulk-seq data files (.csv) | None |
| input_sc_data_path | Path to scRNA-seq data files (.csv) | None |
| input_sc_meta_path | Path to scRNA-seq annotation files (.csv) | None |
| input_st_data_path | Path to ST data files (.csv) | None |
| input_st_meta_path | Path to ST metadata files (.csv) | None |
| ratio_num | The multiples of the number of cells of generated scRNA-seq data | (int) `1` |
| top_marker_num | The number of marker genes of each celltype used | (int) `500`  |
| gpu | The GPU ID. Use cpu if `--gpu < 0` | (int) `0` |
| batch_size | The batch size for β-VAE model training | (int) `512` |
| learning_rate | The learning rate for β-VAE model training | (float) `0.0001` |
| hidden_size | The hidden size of β-VAE model | (int) `256` |
| epoch_num | The epoch number for β-VAE model training | (int) `5000` |
| vae_save_dir | Path to save the trained β-VAE model | (str) `save_model` |
| vae_save_name | File name of the trained β-VAE model | (str) `vae` |
| generate_save_dir | Path to save the generated scRNA-seq data | (str) `output` |
| generate_save_name | File name of the generated scRNA-seq data | (str) `output` |

****
* Decompose spatial barcoding-based spatial transcriptomics data (10x Genomics, ST, or Slide-seq, etc) into spatially resolved single-cell transcriptomics data:
```python
from bulk2space import Bulk2Space
model = Bulk2Space()

# Decompose spatial barcoding-based spatial transcriptomics data 
# (10x Genomics, ST, or Slide-seq, etc) into spatially resolved 
# single-cell transcriptomics data
df_meta, df_data = model.train_df_and_spatial_deconvolution(
    generate_sc_meta,
    generate_sc_data,
    input_st_data_path,
    input_st_meta_path,
    spot_num=500,
    cell_num=10,
    df_save_dir='save_model',
    df_save_name='df',
    map_save_dir='output', 
    map_save_name='deconvolution',
    top_marker_num=500,
    marker_used=True,
    k=10)
```

| Parameter | Description | Default Value |
| --- | --- | --- |
| generate_sc_meta | Generated scRNA-seq metadata | None |
| generate_sc_data | Generated scRNA-seq data | None |
| input_st_data_path | Path to ST data files (.csv) | None |
| input_st_meta_path | Path to ST metadata files (.csv) | None |
| spot_num | The spot number of pseudo-spot data which used to train the deep forest model| (int) `500` |
| cell_num | The cell number per spot of pseudo-spot data which used to train the deep forest model | (int) `10` |
| df_save_dir | Path to save the trained deep forest model | (str) `save_model`  |
| df_save_name | File name of the trained deep forest model | (str) `df` |
| map_save_dir | Path to save the deconvoluted ST data | (str) `output` |
| map_save_name | File name of the deconvoluted ST data | (str) `deconvolution` |
| top_marker_num | The number of marker genes of each celltype used | (int) `500` |
| marker_used | Whether to only use marker genes of each cell type | (bool) `True` |
| k | The number of cells per spot set | (int) `10` |


****
* Map image-based spatial transcriptomics data (MERFISH, SeqFISH, or STARmap, etc) into spatially resolved single-cell transcriptomics data:
```python
from bulk2space import Bulk2Space
model = Bulk2Space()

# Map image-based spatial transcriptomics data (MERFISH, SeqFISH, or STARmap, etc) 
# into spatially resolved single-cell transcriptomics data
df_meta, df_data = model.spatial_mapping(
    generate_sc_meta,
    generate_sc_data,
    input_st_data_path,
    input_st_meta_path)
```


| Parameter | <img width=133/>Description <img width=133/> | <img width=11/>Default Value <img width=11/> |
| --- | --- | --- |
| generate_sc_meta | Generated scRNA-seq metadata | None |
| generate_sc_data | Generated scRNA-seq data | None |
| input_st_data_path | Path to ST data files (.csv) | None |
| input_st_meta_path | Path to ST metadata files (.csv) | None |



