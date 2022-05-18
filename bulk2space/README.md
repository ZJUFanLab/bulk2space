<h1><center>Bulk2Space Tutorial</center></h1>

Jie Liao,  Jingyang Qian, Yin Fang, Zhuo Chen, Xiang Zhuang et al.

## Outline
1. [Installation](#Installation)
2. [Import modules](#Import-modules)
3. [Parameter definition](#Parameter-definition)
4. [Load data](#Load-data)
5. [Marker used](#Marker-used)
6. [Data processing](#Data-processing)
7. [Celltype ratio calculation](#Celltype-ratio-calculation)
8. [Prepare the model input](#Prepare-the-model-input)
9. [Model training/loading](#Model-training/loading)
10. [Data generation](#Data-generation)
11. [Data saving](#Data-saving)


### 1. <a id="Installation">Installation</a>
The installation should take a few minutes on a normal computer. To install Bulk2Space package you must make sure that your python version is over `3.8`. If you don’t know the version of python you can check it by:
```python
import platform
platform.python_version()
```
Note: Because our Bulk2Space dpends on pytorch, you'd better make sure the torch is correctly installed.

### 2. <a id="Import-modules">Import modules</a>

```python
# -*- coding: utf-8 -*-
from utils.tool import *
from utils.config import cfg, loadArgums
import numpy as np
import pandas as pd
import torch
import scanpy
from scipy.optimize import nnls
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')
```


### 3. <a id="Parameter-definition">Parameter definition</a>
For the current version of Bulk2Space,

some parameters should be revised  according to the actual running environment and file Hierarchy:

- `gpu_id`: The GPU ID, eg:`--gpu_id 0`
- `project_name`: The name of your project, eg:`--project_name experiment1`
- `input_bulk_path`: The name of the input bulk-seq data, eg:`--input_bulk_path bulk_data.csv`
- `input_sc_data_path`: The name of the input scRNA-seq data, eg:`--input_sc_data_path sc_data.csv`
- `input_sc_meta_path`: The name of the input scRNA-seq meta, eg:`--input_sc_meta_path sc_meta.csv`
- `input_st_data_path`: The name of the input spatial transcriptomics data, eg:`--input_st_data_path st_data.csv`
- `input_st_meta_path`: The name of the input spatial transcriptomics meta, eg:`--input_st_meta_path st_meta.csv`
- `load_model_1`: Whether to load the trained bulk-deconvolution model, eg:`--load_model_1 False`
- `load_path_1`: The path of the trained bulk-deconvolution model to be loaded
- `train_model_2`: Whether to train the spatial mapping model, eg:`--train_model_2 True`
- `load_path_2`: The path of the trained spatial mapping model to be loaded
- `output_path`: The name of the folder where you store the output data, eg:`--output_path output_data`


some parameters could be revised  as needed:
- `BetaVAE_H`: Whether to use β-VAE model or not, eg:`--BetaVAE_H`
- `batch_size`: The batch size for β-VAE/VAE model training, eg:`--batch_size 512`
- `learning_rate`: The learning rate for β-VAE/VAE model training, eg:`--learning_rate 0.0001`
- `hidden_size`: The hidden size of β-VAE/VAE model, eg:`--hidden_size 256`
- `hidden_lay`: The hidden layer of β-VAE/VAE model(0:[2048, 1024, 512] \n 1: [4096, 2048, 1024, 512] \n 2: [8192, 4096, 2048, 1024]), eg:`--hidden_lay 0`
- `epoch_num`: The epoch number for β-VAE/VAE model training, eg:`--epoch_num 5000`
- `not_early_stop`: Whether to use the `early_stop` strategy, eg:`--not_early_stop False`
- `early_stop`: The model waits N epochs before stops if no progress on the validation set or the training loss dose not decline, eg:`--early_stop 50`
- `k`: The number of cells per spot set in spatial mapping step, eg:`--k 10`
- `marker_used`: Whether to only use marker genes of each celltype when calculating the celltype proportion, eg:`--marker_used True`
- `top_marker_num`: The number of marker genes of each celltype used, eg:`--top_marker_num 500`
- `ratio_num`: The multiples of the number of cells of generated scRNA-seq data, eg:`--ratio_num 1`
- `spot_data`: The type of the input spatial transcriptomics data, `True` for barcoded-based ST data (like ST, 10x Visium or Slide-seq) and  `False` for image-based ST data (like MERFISH, SeqFISH or STARmap)

```python
global args 
args = dict(
    BetaVAE_H=True,
    batch_size=512,
    cell_num=10,
    data_path='example_data/demo1',
    dump_path='/data/zhuangxiang/code/bulk2space/bulk2space/dump',
    early_stop=50,
    epoch_num=10,
    exp_id='LR_0.0001_hiddenSize_256_lay_choice_0',
    exp_name='test1',
    feature_size=6588,
    gpu_id=0,
    hidden_lay=0,
    hidden_size=256,
    input_bulk_path='/data/zhuangxiang/code/bulk2space/bulk2space/data/example_data/demo1/demo1_bulk.csv',
    input_sc_data_path='/data/zhuangxiang/code/bulk2space/bulk2space/data/example_data/demo1/demo1_sc_data.csv',
    input_sc_meta_path='/data/zhuangxiang/code/bulk2space/bulk2space/data/example_data/demo1/demo1_sc_meta.csv',
    input_st_data_path='/data/zhuangxiang/code/bulk2space/bulk2space/data/example_data/demo1/demo1_st_data.csv',
    input_st_meta_path='/data/zhuangxiang/code/bulk2space/bulk2space/data/example_data/demo1/demo1_st_meta.csv',
    k=10, 
    kl_loss=False, 
    learning_rate=0.0001, 
    load_model_1=False, 
    load_path_1='/data/zhuangxiang/code/bulk2space/bulk2space/save_model/',
    load_path_2='/data/zhuangxiang/code/bulk2space/bulk2space/save_model/', marker_used=True, 
    max_cell_in_diff_spot_ratio=None, 
    model_choice_1='vae', 
    model_choice_2='df',
    mul_test=5, 
    mul_train=1, 
    no_tensorboard=False, 
    not_early_stop=False, 
    num_workers=12, 
    output_path='/data/zhuangxiang/code/bulk2space/bulk2space/output_data',
    previous_project_name='demo', 
    project_name='test1', 
    random_seed=12345, 
    ratio_num=1,
    save='/data/zhuangxiang/code/bulk2space/bulk2space/save_model', 
    spot_data=True, 
    spot_num=500,
    top_marker_num=500, 
    train_model_2=True, 
    xtest='xtest', 
    xtrain='xtrain', 
    ytest='ytest', 
    ytrain='ytrain'
)
args = argparse.Namespace(**args)
```

```python
input_sc_meta_path = args.input_sc_meta_path
input_sc_data_path = args.input_sc_data_path
input_bulk_path = args.input_bulk_path
input_st_meta_path = args.input_st_meta_path
input_st_data_path = args.input_st_data_path
```

### 4. <a id="Load-data">Load data</a>
`Bulk2Space` requires five formatted data as input:
- Bulk-seq Normalized Data
    - a `.csv` file with genes as rows and sample as column
        |  | Sample | 
    | ----- | ----- | 
    | Gene1 | 5.22 |
    | Gene2 | 3.67 |
    | ... | ... |
    | GeneN | 15.76 |
- Single Cell RNA-seq Normalized Data
    - a `.csv` file with genes as rows and cells as columns
- Single Cell RNA-seq Annotation Data
    - a `.csv` file with cell names and celltype annotation columns. The column containing cell names should be named `Cell` and the column containing the labels should be named `Cell_type`
- Spatial Transcriptomics Normalized Data
    - a `.csv` file with genes as rows and cells/spots as columns
- Spatial Transcriptomics Coordinates Data
    - a `.csv` with cell/spot names and coordinates columns. The column containing cell/spot names should be named `Spot` and the column containing the coordinates should be named `xcoord` and `ycoord`

```python
print("loading data......")

# load sc_meta.csv file, containing two columns of cell name and cell type
input_sc_meta = pd.read_csv(input_sc_meta_path, index_col=0)
# load sc_data.csv file, containing gene expression of each cell
input_sc_data = pd.read_csv(input_sc_data_path, index_col=0)
sc_gene = input_sc_data._stat_axis.values.tolist()
# load bulk.csv file, containing one column of gene expression in bulk
input_bulk = pd.read_csv(input_bulk_path, index_col=0)
bulk_gene = input_bulk._stat_axis.values.tolist()
# filter overlapping genes.
intersect_gene = list(set(sc_gene).intersection(set(bulk_gene)))
input_sc_data = input_sc_data.loc[intersect_gene]
input_bulk = input_bulk.loc[intersect_gene]
# load st_meta.csv and st_data.csv, containing coordinates and gene expression of each spot respectively.
input_st_meta = pd.read_csv(input_st_meta_path, index_col=0)
input_st_data = pd.read_csv(input_st_data_path, index_col=0)
print("load data ok")
```

### 5. <a id="Marker-used">Marker used</a>
```python
# marker used
sc = scanpy.AnnData(input_sc_data.T)
sc.obs = input_sc_meta[['Cell_type']]
scanpy.tl.rank_genes_groups(sc, 'Cell_type', method='wilcoxon')
marker_df = pd.DataFrame(sc.uns['rank_genes_groups']['names']).head(args.top_marker_num)
marker_array = np.array(marker_df)
marker_array = np.ravel(marker_array)
marker_array = np.unique(marker_array)
marker = list(marker_array)
sc_marker = input_sc_data.loc[marker, :]
bulk_marker = input_bulk.loc[marker]
```

### 6. <a id="Data-processing">Data processing</a>
```python
breed = input_sc_meta['Cell_type']
breed_np = breed.values
breed_set = set(breed_np)
id2label = sorted(list(breed_set))  # List of breed
label2id = {label: idx for idx, label in enumerate(id2label)}  # map breed to breed-id

cell2label = dict()  # map cell-name to breed-id
label2cell = defaultdict(set)  # map breed-id to cell-names
for row in input_sc_meta.itertuples():
    cell_name = getattr(row, 'Cell')
    cell_type = label2id[getattr(row, 'Cell_type')]
    cell2label[cell_name] = cell_type
    label2cell[cell_type].add(cell_name)

label_devide_data = dict()
for label, cells in label2cell.items():
    label_devide_data[label] = sc_marker[list(cells)]

single_cell_splitby_breed_np = {}
for key in label_devide_data.keys():
    single_cell_splitby_breed_np[key] = label_devide_data[key].values  # [gene_num, cell_num]
    single_cell_splitby_breed_np[key] = single_cell_splitby_breed_np[key].mean(axis=1)

max_decade = len(single_cell_splitby_breed_np.keys())
single_cell_matrix = []

for i in range(max_decade):
    single_cell_matrix.append(single_cell_splitby_breed_np[i].tolist())


single_cell_matrix = np.array(single_cell_matrix)
single_cell_matrix = np.transpose(single_cell_matrix)  # (gene_num, label_num)

bulk_marker = bulk_marker.values  # (gene_num, 1)
bulk_rep = bulk_marker.reshape(bulk_marker.shape[0],)
```

### 7.  <a id="Celltype-ratio-calculation">Celltype ratio calculation</a>
```python
# calculate celltype ratio in each spot by NNLS
ratio = nnls(single_cell_matrix, bulk_rep)[0]
ratio = ratio/sum(ratio)
ratio_array = np.round(ratio * input_sc_meta.shape[0] * args.ratio_num)
ratio_list = [r for r in ratio_array]
cell_target_num = dict(zip(id2label, ratio_list))
```

### 8. <a id="Prepare-the-model-input">Prepare the model input</a>
```python
# *********************************************************************
# input：data， celltype， bulk & output: label, dic, single_cell
single_cell = input_sc_data.values.T  
index_2_gene = (input_sc_data.index).tolist()
breed = input_sc_meta['Cell_type']
breed_np = breed.values
breed_set = set(breed_np)
breed_2_list = list(breed_set)
dic = {}
label = []  # the label of cell (with index correspond)
cfg.nclass = len(breed_set)

cfg.ntrain = single_cell.shape[0]
cfg.FeaSize = single_cell.shape[1]
args.feature_size = single_cell.shape[1]
assert cfg.nclass == len(cell_target_num.keys()), "cell type num no match!!!"

for i in range(len(breed_set)):
    dic[breed_2_list[i]] = i
cell = input_sc_meta["Cell"].values

for i in range(cell.shape[0]):
    label.append(dic[breed_np[i]])

label = np.array(label)

# label index the data size of corresponding target
cell_number_target_num = {}
for k, v in cell_target_num.items():
    cell_number_target_num[dic[k]] = v
# *********************************************************************
# generate data by vae
load_model_1 = args.load_model_1
model_choice_1 = args.model_choice_1
```

### 9. <a id="Model-training/loading">Model training/loading</a>
```python
logger = initialize_exp(args)

ratio = -1
if not load_model_1:  # train
    logger.info("begin vae model training...")
    # ********************* training *********************
    net = train_vae(args, single_cell, cfg, label)
    # ************** training finished *******************
    logger.info("vae training finished!")
else:  # load model
    logger.info("begin vae model loading...")
    net = load_vae(args, cfg)
    logger.info("vae load finished!")
```

### 10. <a id="Data-generation">Data generation</a>
```python
# generate and out put
generate_sc_meta, generate_sc_data = generate_vae(net, args, ratio, single_cell, cfg, label, breed_2_list, index_2_gene, cell_number_target_num)
```

### 11. <a id="Data-saving">Data saving</a>
```python
# saving.....
path = osp.join(args.output_path, args.project_name, 'predata')
if not osp.exists(path):
    os.makedirs(path)
name = "vae"
# kl_loss BetaVAE_H
if args.BetaVAE_H:
    name = "BetaVAE"
path_label_generate_csv = os.path.join(path, args.project_name + "_celltype_pred_" + name + "_epoch" + str(args.epoch_num) + '_lr' + str(args.learning_rate) + ".csv")
path_cell_generate_csv = os.path.join(path, args.project_name + "_data_pred_" + name + "_epoch" + str(args.epoch_num) + '_lr' + str(args.learning_rate) + ".csv")

generate_sc_meta.to_csv(path_label_generate_csv)
generate_sc_data.to_csv(path_cell_generate_csv)

print("bulk deconvolution finish!")
```
