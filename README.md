# Bulk2Space

## De novo analysis of bulk RNA-seq data at spatially resolved single-cell resolution 
### Jie Liao<sup>†</sup>,  Jingyang Qian<sup>†</sup>, Yin Fang<sup>†</sup>, Zhuo Chen<sup>†</sup>, Xiang Zhuang<sup>†</sup>, ..., Huajun Chen\*, Xiaohui Fan*

[![python 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/) 

Bulk2Space is a two-step spatial deconvolution method based on deep learning frameworks, which converts bulk transcriptomes into spatially resolved single-cell expression profiles.

![Image text](images/overview.jpeg)

# Requirements and Installation
[![tensorboard 2.6.0](https://img.shields.io/badge/tensorboard-2.6.0-brightgreen)](https://pypi.org/project/tensorboard/) [![numpy 1.19.2](https://img.shields.io/badge/numpy-1.19.2-green)](https://github.com/numpy/numpy) [![pandas 1.1.3](https://img.shields.io/badge/pandas-1.1.3-yellowgreen)](https://github.com/pandas-dev/pandas) [![scikit-learn 1.0.1](https://img.shields.io/badge/scikit--learn-1.0.1-yellow)](https://github.com/scikit-learn/scikit-learn) [![scipy 1.5.2](https://img.shields.io/badge/scipy-1.5.2-orange)](https://github.com/scipy/scipy) [![torch 1.7.0](https://img.shields.io/badge/torch-1.7.0-red)](https://github.com/pytorch/pytorch) [![deep-forest 0.1.5](https://img.shields.io/badge/deep--forest-0.1.5-success)](https://pypi.org/project/deep-forest/) [![easydict 1.9](https://img.shields.io/badge/easydict-1.9-informational)](https://pypi.org/project/easydict/) [![pytorch-transformers 1.2.0](https://img.shields.io/badge/pytorch--transformers-1.2.0-blueviolet)](https://pypi.org/project/pytorch-transformers/) [![scanpy 1.8.1](https://img.shields.io/badge/scanpy-1.8.1-ff69b4)](https://pypi.org/project/scanpy/) [![tqdm 4.50.2](https://img.shields.io/badge/tqdm-4.50.2-9cf)](https://pypi.org/project/tqdm/) [![Unidecode 1.3.0](https://img.shields.io/badge/Unidecode-1.3.0-inactive)](https://pypi.org/project/Unidecode/) 

For bulk2space, the python version need is over 3.8. If you have installed Python3.6 or Python3.7, consider installing Anaconda, and then you can create a new environment.
```
conda create -n bulk2space python=3.8.5
conda activate bulk2space

cd bulk2space
pip install -r requirements.txt 
```

# Usage

## Run the demo data
If you choose the spatial barcoding-based data (10x Genomics, ST, or Slide-seq) as spatial reference, run the following command:
```
python bulk2space.py --project_name test1 --data_path example_data/demo1 --input_sc_meta_path demo1_sc_meta.csv --input_sc_data_path demo1_sc_data.csv --input_bulk_path demo1_bulk.csv --input_st_data_path demo1_st_data.csv --input_st_meta_path demo1_st_meta.csv --BetaVAE_H --epoch 3000 --spot_data True
```

else, if you choose the image-based in situ hybridization data (MERFISH, SeqFISH, or STARmap) as spatial reference, run the following command:
```
python bulk2space.py --project_name test2 --data_path example_data/demo2 --input_sc_meta_path demo2_sc_meta.csv --input_sc_data_path demo2_sc_data.csv --input_bulk_path demo2_bulk.csv --input_st_data_path demo2_st_data.csv --input_st_meta_path demo2_st_meta.csv --BetaVAE_H --epoch 3000 --spot_data False
```

## Run your own data
`Bulk2Space` requires five formatted data as input when using your own data:
* Bulk-seq Normalized Data
  * a `.csv` file with genes as rows and sample as column
        
      |<img width=120/> <img width=120/>|<img width=120/>Sample<img width=120/>| 
      | :-----: | :-----: | 
      | Gene1 | 5.22 |
      | Gene2 | 3.67 |
      | ... | ... |
      | GeneN | 15.76 |
  
* Single Cell RNA-seq Normalized Data
  * a `.csv` file with genes as rows and cells as columns
  
      |<img width=30/> <img width=30/>|<img width=20/>Cell1<img width=20/>|<img width=20/>Cell2<img width=20/>|<img width=20/>Cell3<img width=20/>|<img width=20/>...<img width=25/>|<img width=25/>CellN<img width=20/>| 
      | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
      | Gene1 | 1.05 | 2.31 | 1.72 | ... | 0 |
      | Gene2 | 4.71 | 1.07 | 0 | ... | 4.22 |
      | ... | ... | ... | ... | ... | ... |
      | GeneN | 0.55 | 0 | 1.48 | ... | 0 |

* Single Cell RNA-seq Annotation Data
  * a `.csv` file with cell names and celltype annotation columns. The column containing cell names should be named `Cell` and the column containing the labels should be named `Cell_type` 

      |<img width=65/> <img width=65/>|<img width=70/>Cell<img width=70/>|<img width=70/>Cell_type<img width=70/>|
      | :-----: | :-----: | :-----: |
      | Cell1 | Cell1 | T cell |
      | Cell2 | Cell2 | B cell |
      | ... | ... | ... |
      | CellN | CellN | Monocyte |

* Spatial Transcriptomics Normalized Data
  * a `.csv` file with genes as rows and cells/spots as columns 

      |<img width=20/> <img width=20/>|<img width=10/>Cell1 / Spot1<img width=10/>|<img width=10/>Cell2 / Spot2<img width=20/>|<img width=30/>...<img width=30/>|<img width=10/>CellN / SpotN<img width=10/>| 
      | :-----: | :-----: | :-----: | :-----: | :-----: |
      | Gene1 | 3.22 | 4.71 | ... | 1.01 |
      | Gene2 | 0 | 2.17 | ... | 2.20 |
      | ... | ... | ... | ... | ... |
      | GeneN | 0 | 0.11 | ... | 1.61 |

* Spatial Transcriptomics Coordinates Data
  * a `.csv` with cell/spot names and coordinates columns. The column containing cell/spot names should be named `Spot` and the column containing the coordinates should be named `xcoord` and `ycoord`

    |<img width=50/> <img width=50/>|<img width=40/>Spot<img width=40/>|<img width=40/>xcoord<img width=40/>|<img width=40/>ycoord<img width=40/>|
    | :-----: | :-----: | :-----: | :-----: |
    | Cell_1 / Spot_1 | Cell_1 / Spot_1 | 1.2 | 5.2 |
    | Cell_2 / Spot_2 | Cell_1 / Spot_1 |5.4 | 4.3 |
    | ... | ... | ... | ... |
    | Cell_n / Spot_n | Cell_1 / Spot_1 | 11.3 | 6.3 |
  
Then you will get your results in the `output_data` folder.

# Tutorials
**The step-by-step tutorial now available!**

To figure out the important parameters of Bulk2Space or perform a test run, please refer to:
* [Demonstration of Bulk2Space on demo data](bulk2space/tutorial/demo.ipynb)

To perform Bulk2Space with the barcoding-based spatial reference (10x Genomics, ST, or Slide-seq):

* [Integrating spatial gene expression and histomorphology in pancreatic ductal adenocarcinoma (PDAC)](bulk2space/tutorial/PDAC_analysis.ipynb)

To perform Bulk2Space with the image-based spatial reference (MERFISH, SeqFISH, or STARmap) (TODO):

* [Re-annotates ambiguous cells in the mouse hypothalamus]()

# About
Should you have any questions, please feel free to contact the co-first authors of the manuscript, Dr. Jie Liao (liaojie@zju.edu.cn), Mr. Jingyang Qian (qianjingyang@zju.edu.cn), Miss Yin Fang (fangyin@zju.edu.cn), Mr. Zhuo Chen (zhuo.chen@zju.edu.cn), or Mr. Xiang Zhuang (zhuangxiang@zju.edu.cn)
