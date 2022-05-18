# -*- coding: utf-8 -*-
from utils.tool import *
from utils.config import cfg, loadArgums
from torchlight import initialize_exp, set_seed, snapshot, get_dump_path, show_params
import numpy as np
import pandas as pd
import torch
import scanpy
from scipy.optimize import nnls
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


def main():
    print("***************path change*****************")
    global args
    args = loadArgums(cfg)
    used_device = torch.device(
        f"cuda:{args.gpu_id}") if args.gpu_id >= 0 and torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.set_device(args.gpu_id)
    print("***************bulk to spatial*****************")

    input_sc_meta_path = args.input_sc_meta_path
    input_sc_data_path = args.input_sc_data_path
    input_bulk_path = args.input_bulk_path
    input_st_meta_path = args.input_st_meta_path
    input_st_data_path = args.input_st_data_path
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
    bulk_rep = bulk_marker.reshape(bulk_marker.shape[0], )

    # calculate celltype ratio in each spot by NNLS
    ratio = nnls(single_cell_matrix, bulk_rep)[0]
    ratio = ratio / sum(ratio)

    ratio_array = np.round(ratio * input_sc_meta.shape[0] * args.ratio_num)
    ratio_list = [r for r in ratio_array]

    cell_target_num = dict(zip(id2label, ratio_list))

    # *********************************************************************
    # input：data， celltype， bulk & output: label, dic, single_cell
    single_cell = input_sc_data.values.T  # single cell data (600 * 6588)
    index_2_gene = (input_sc_data.index).tolist()
    breed = input_sc_meta['Cell_type']
    breed_np = breed.values
    breed_set = set(breed_np)
    breed_2_list = list(breed_set)
    dic = {}  # breed_set to index {'B cell': 0, 'Monocyte': 1, 'Dendritic cell': 2, 'T cell': 3}
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

    logger = initialize_exp(args)

    ratio = -1
    if model_choice_1 == "vae":
        if not load_model_1:  # train
            logger.info("begin vae model training...")
            # ********************* training *********************
            net = train_vae(args, single_cell, cfg, label, used_device)
            # ************** training finished *******************
            logger.info("vae training finished!")
        else:  # load model
            logger.info("begin vae model loading...")
            net = load_vae(args, cfg, used_device)
            logger.info("vae load finished!")

        # generate and out put
        generate_sc_meta, generate_sc_data = generate_vae(net, args, ratio, single_cell, cfg, label, breed_2_list,
                                                          index_2_gene, cell_number_target_num, used_device)
        # saving.....
        path = osp.join(args.output_path, args.project_name, 'predata')
        if not osp.exists(path):
            os.makedirs(path)
        name = "vae"
        # kl_loss BetaVAE_H
        if args.BetaVAE_H:
            name = "BetaVAE"
        path_label_generate_csv = os.path.join(path, args.project_name + "_celltype_pred_" + name + "_epoch" + str(
            args.epoch_num) + '_lr' + str(args.learning_rate) + ".csv")
        path_cell_generate_csv = os.path.join(path, args.project_name + "_data_pred_" + name + "_epoch" + str(
            args.epoch_num) + '_lr' + str(args.learning_rate) + ".csv")

        generate_sc_meta.to_csv(path_label_generate_csv)
        generate_sc_data.to_csv(path_cell_generate_csv)

        logger.info("bulk deconvolution finish!")

        print('start to map data to space...')

        # TODO: MERFISH Data mapping
        if args.spot_data:
            processer = CreatData(generate_sc_data, generate_sc_meta, input_st_data, args)
            processer.cre_data()

            # cre_data(args, generate_sc_data, generate_sc_meta)

            runner = Runner(generate_sc_data, generate_sc_meta, input_st_data, input_st_meta, args)
            print('start to train the model...')
            runner.run()
            logger.info("spatial mapping done!")
        else:
            print('start to process MERFISH-like st data...')
            sc_gene_2 = generate_sc_data._stat_axis.values.tolist()
            st_gene_2 = input_st_data._stat_axis.values.tolist()
            intersect_gene_2 = list(set(sc_gene_2).intersection(set(st_gene_2)))
            generate_sc_data_2 = generate_sc_data.loc[intersect_gene_2]
            input_st_data_2 = input_st_data.loc[intersect_gene_2]

            sc_cell_rename = [f'SC_{i}' for i in range(1, generate_sc_data_2.shape[1] + 1)]
            generate_sc_data.columns = generate_sc_data_2.columns = sc_cell_rename
            generate_sc_meta = generate_sc_meta.drop(['Cell'], axis=1)
            generate_sc_meta.insert(0, 'Cell', sc_cell_rename)
            generate_sc_meta['Batch'] = 'sc'
            generate_sc_meta_2 = generate_sc_meta.drop(['Cell_type'], axis=1)
            st_cell_rename = [f'ST_{i}' for i in range(1, input_st_data_2.shape[1] + 1)]
            input_st_data.columns = input_st_data_2.columns = st_cell_rename
            input_st_meta = input_st_meta.drop(['Cell'], axis=1)
            input_st_meta.insert(0, 'Cell', st_cell_rename)
            input_st_meta_2 = pd.DataFrame({'Cell': st_cell_rename, 'Batch': 'st'})

            all_data = generate_sc_data_2.join(input_st_data_2)
            all_meta = pd.concat([generate_sc_meta_2, input_st_meta_2], ignore_index=True)
            joint_data = joint_analysis(all_data, all_meta['Batch'], ref_batch="st")
            joint_data[joint_data < 0] = 0
            sc_data_new = joint_data.iloc[:, 0:generate_sc_data_2.shape[1]]
            st_data_new = joint_data.iloc[:, generate_sc_data_2.shape[1]:all_data.shape[1]]

            _, ind = knn(data=sc_data_new.T, query=st_data_new.T, k=10)

            st_data_pred = pd.DataFrame()
            st_meta_pred = pd.DataFrame(columns=['Cell', 'Cell_type'])

            for i in range(len(st_cell_rename)):
                st_data_pred[st_cell_rename[i]] = list(generate_sc_data.iloc[:, ind[i]].mean(axis=1))
                ct_tmp = list(generate_sc_meta.iloc[ind[i], :].Cell_type)
                ct_pred = max(ct_tmp, key=ct_tmp.count)
                st_meta_pred.loc[st_cell_rename[i]] = [st_cell_rename[i], ct_pred]

            st_data_pred.index = generate_sc_data.index
            st_meta_pred = pd.merge(st_meta_pred, input_st_meta, how='left', on='Cell')

            #  save df
            os.makedirs(args.output_path, exist_ok=True)

            st_meta_pred.to_csv(os.path.join(args.output_path, args.project_name, f'meta_{args.project_name}.csv'))
            st_data_pred.to_csv(os.path.join(args.output_path, args.project_name, f'data_{args.project_name}.csv'))
            print('save csv ok')

    else:
        print("model choice error!!!")
        exit()


if __name__ == "__main__":
    main()
