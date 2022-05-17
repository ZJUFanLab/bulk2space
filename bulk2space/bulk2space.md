<h1><center>Bulk2space Tutorial</center></h1>

Jie Liao*,  Jingyang Qian, Yin Fang, Zhuo Chen, Xiang Zhuang

## Outline
1. [Installation](tutorial.md#1-installation)
2. [Import modules](tutorial.md#2-import-modules)
3. [Read in data](tutorial.md#3-read-in-data)
4. [Integrate gene expression and histology into a Graph](tutorial.md#4-integrate-gene-expression-and-histology-into-a-graph)
5. [Spatial domain detection using SpaGCN](tutorial.md#5-spatial-domain-detection-using-spagcn)
6. [Identify SVGs](tutorial.md#6-identify-svgs)
7. [Identify Meta Gene](tutorial.md#7-identify-meta-gene)
8. [Multiple tissue sections analysis](tutorial.md#8-multiple-tissue-sections-analysis)

### 1. Installation
The installation should take a few minutes on a normal computer. To install SpaGCN package you must make sure that your python version is over `3.8`. If you don’t know the version of python you can check it by:
```python
import platform
platform.python_version()
```
Note: Because our bulk2space dpends on pytorch, you'd better make sure the torch is correctly installed.

### 2. Import modules

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


### 3. Parameter definition
For the current version of SpaGCN,

some parameters should be revised  according to the actual running environment and file Hierarchy:

- gpu_id
- feature_size
- data_path
- dump_path
- input_bulk_path
- input_sc_data_path
- input_sc_meta_path
- input_st_data_path
- input_st_meta_path
- load_path_1
- load_path_2
- output_path
- previous_project_name
- project_name

some parameters could be revised  as needed:
- BetaVAE_H
- batch_size
- cell_num
- early_stop
- epoch_num
- exp_id
- exp_name
- hidden_lay
- hidden_size
- k
- kl_loss
- learning_rate
- load_model_1
- marker_used
- max_cell_in_diff_spot_ratio
- model_choice_1
- model_choice_2
- mul_test
- mul_train
- no_tensorboard
- not_early_stop
- num_workers
- random_seed
- ratio_num
- spot_data
- spot_num
- top_marker_num
- train_model_2

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

### 4. Load data
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

### 4. Marker used
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

### 5. Data processing
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
#
for i in range(max_decade):
    single_cell_matrix.append(single_cell_splitby_breed_np[i].tolist())


single_cell_matrix = np.array(single_cell_matrix)
single_cell_matrix = np.transpose(single_cell_matrix)  # (gene_num, label_num)

bulk_marker = bulk_marker.values  # (gene_num, 1)
bulk_rep = bulk_marker.reshape(bulk_marker.shape[0],)
```

### 5.  Celltype ratio calculation
```python
# calculate celltype ratio in each spot by NNLS
ratio = nnls(single_cell_matrix, bulk_rep)[0]
ratio = ratio/sum(ratio)
ratio_array = np.round(ratio * input_sc_meta.shape[0] * args.ratio_num)
ratio_list = [r for r in ratio_array]
cell_target_num = dict(zip(id2label, ratio_list))
```

### 6. Prepare the model input
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

### 6. Model training/loading
```python
logger = initialize_exp(args)
# logger_path = get_dump_path(args)


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

### 7. Data generation
```python
    # generate and out put
generate_sc_meta, generate_sc_data = generate_vae(net, args, ratio, single_cell, cfg, label, breed_2_list, index_2_gene, cell_number_target_num)
```

### 8. Data saving
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