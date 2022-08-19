import pandas as pd
import torch
import numpy as np
from .utils import load_data, data_process
from .vae import train_vae, generate_vae, load_vae
from .map_utils import create_data, DFRunner, joint_analysis, knn

import os
import warnings

warnings.filterwarnings("ignore")


class Bulk2Space:
    def __init__(self):
        pass

    def train_vae_and_generate(self,
                               input_bulk_path,
                               input_sc_data_path,
                               input_sc_meta_path,
                               input_st_data_path,
                               input_st_meta_path,
                               ratio_num=1,
                               # marker_used,
                               top_marker_num=500,
                               vae_save_dir='save_model',
                               vae_save_name='vae',
                               generate_save_dir='output',
                               generate_save_name='output',
                               gpu=0,
                               batch_size=512,
                               learning_rate=1e-4,
                               hidden_size=256,
                               epoch_num=5000):
        used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        input_data = load_data(input_bulk_path,
                               input_sc_data_path,
                               input_sc_meta_path,
                               input_st_data_path,
                               input_st_meta_path)
        cell_target_num = data_process(input_data, top_marker_num, ratio_num)
        single_cell, label, breed_2_list, index_2_gene, cell_number_target_num, \
        nclass, ntrain, feature_size = self.__get_model_input(input_data, cell_target_num)
        print('begin vae training...')
        vae_net = train_vae(single_cell,
                            label,
                            used_device,
                            batch_size,
                            feature_size=feature_size,
                            epoch_num=epoch_num,
                            learning_rate=learning_rate,
                            hidden_size=hidden_size)
        print('vae training done!')
        path_save = os.path.join(vae_save_dir, f"{vae_save_name}.pth")
        if not os.path.exists(vae_save_dir):
            os.makedirs(vae_save_dir)
        torch.save(vae_net.state_dict(), path_save)
        print(f"save trained vae in {path_save}.")
        print('generating....')
        generate_sc_meta, generate_sc_data = generate_vae(vae_net, -1,
                                                          single_cell, label, breed_2_list,
                                                          index_2_gene, cell_number_target_num, used_device)
        self.__save_generation(generate_sc_meta, generate_sc_data,
                               generate_save_dir, generate_save_name)
        return generate_sc_meta, generate_sc_data

    def load_vae_and_generate(self,
                              input_bulk_path,
                              input_sc_data_path,
                              input_sc_meta_path,
                              input_st_data_path,
                              input_st_meta_path,
                              vae_load_dir,  # load_dir
                              ratio_num=1,
                              # marker_used,
                              top_marker_num=500,
                              generate_save_dir='output',  # file_dir
                              generate_save_name='generation',  # file_name
                              gpu=0,
                              hidden_size=256):
        used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        input_data = load_data(input_bulk_path,
                               input_sc_data_path,
                               input_sc_meta_path,
                               input_st_data_path,
                               input_st_meta_path)
        cell_target_num = data_process(input_data, top_marker_num, ratio_num)
        single_cell, label, breed_2_list, index_2_gene, cell_number_target_num, \
        nclass, ntrain, feature_size = self.__get_model_input(input_data, cell_target_num)
        print(f'loading model from {vae_load_dir}')
        vae_net = load_vae(feature_size, hidden_size, vae_load_dir, used_device)
        print('generating....')
        generate_sc_meta, generate_sc_data = generate_vae(vae_net, -1,
                                                          single_cell, label, breed_2_list,
                                                          index_2_gene, cell_number_target_num, used_device)
        self.__save_generation(generate_sc_meta, generate_sc_data, generate_save_dir, generate_save_name)
        print('generating done!')
        return generate_sc_meta, generate_sc_data

    def train_df_and_spatial_deconvolution(self,
                                           generate_sc_meta,
                                           generate_sc_data,
                                           input_st_data_path,
                                           input_st_meta_path,
                                           spot_num,
                                           cell_num,
                                           df_save_dir='save_model',
                                           df_save_name='df',
                                           map_save_dir='output',  # file_dir
                                           map_save_name='deconvolution',  # file_name
                                           top_marker_num=500,
                                           marker_used=True,
                                           max_cell_in_diff_spot_ratio=None,
                                           k=10,
                                           mul_train=1,
                                           random_seed=0):
        input_st_data = pd.read_csv(input_st_data_path, index_col=0)
        input_st_meta = pd.read_csv(input_st_meta_path, index_col=0)
        xtrain, ytrain = create_data(generate_sc_meta, generate_sc_data, input_st_data, spot_num, cell_num,
                                     top_marker_num,
                                     marker_used, mul_train)
        df_runner = DFRunner(generate_sc_data, generate_sc_meta, input_st_data, input_st_meta,
                             marker_used, top_marker_num, random_seed=random_seed)
        df_meta, df_spot = df_runner.run(xtrain, ytrain, max_cell_in_diff_spot_ratio, k, df_save_dir, df_save_name)
        #  save df
        os.makedirs(map_save_dir, exist_ok=True)
        meta_dir = os.path.join(map_save_dir, f'meta_{map_save_name}_{k}.csv')
        spot_dir = os.path.join(map_save_dir, f'data_{map_save_name}_{k}.csv')
        df_meta.to_csv(meta_dir)
        df_spot.to_csv(spot_dir)
        print(f"saving result to {meta_dir} and {spot_dir}")
        return df_meta, df_spot

    def load_df_and_spatial_deconvolution(self,
                                          generate_sc_meta,
                                          generate_sc_data,
                                          input_st_data_path,
                                          input_st_meta_path,
                                          spot_num,
                                          cell_num,
                                          df_load_dir='save_model/df',
                                          map_save_dir='output',  # file_dir
                                          map_save_name='deconvolution',  # file_name
                                          top_marker_num=500,
                                          marker_used=True,
                                          max_cell_in_diff_spot_ratio=None,
                                          k=10,
                                          mul_train=1):
        input_st_data = pd.read_csv(input_st_data_path, index_col=0)
        input_st_meta = pd.read_csv(input_st_meta_path, index_col=0)
        xtrain, ytrain = create_data(generate_sc_meta, generate_sc_data, input_st_data, spot_num, cell_num,
                                     top_marker_num,
                                     marker_used, mul_train)
        df_runner = DFRunner(generate_sc_data, generate_sc_meta, input_st_data, input_st_meta,
                             marker_used, top_marker_num)
        df_meta, df_spot = df_runner.run(xtrain, ytrain, max_cell_in_diff_spot_ratio, k, None, None, df_load_dir)
        #  save df
        os.makedirs(map_save_dir, exist_ok=True)
        meta_dir = os.path.join(map_save_dir, f'meta_{map_save_name}_{k}.csv')
        spot_dir = os.path.join(map_save_dir, f'data_{map_save_name}_{k}.csv')
        df_meta.to_csv(meta_dir)
        df_spot.to_csv(spot_dir)
        print(f"saving result to {meta_dir} and {spot_dir}")
        return df_meta, df_spot

    def spatial_mapping(self,
                        generate_sc_meta,
                        generate_sc_data,
                        input_st_data_path,
                        input_st_meta_path,
                        map_save_dir='output',  # file_dir
                        map_save_name='map',  # file_name
                        ):
        input_st_data = pd.read_csv(input_st_data_path, index_col=0)
        input_st_meta = pd.read_csv(input_st_meta_path, index_col=0)
        print('start to process image-based st data...')
        sc_gene_new = generate_sc_data._stat_axis.values.tolist()
        st_gene_new = input_st_data._stat_axis.values.tolist()
        intersect_gene_new = list(set(sc_gene_new).intersection(set(st_gene_new)))
        generate_sc_data_new = generate_sc_data.loc[intersect_gene_new]
        input_st_data_new = input_st_data.loc[intersect_gene_new]

        sc_cell_rename = [f'SC_{i}' for i in range(1, generate_sc_data_new.shape[1] + 1)]
        generate_sc_data.columns = generate_sc_data_new.columns = sc_cell_rename
        generate_sc_meta = generate_sc_meta.drop(['Cell'], axis=1)
        generate_sc_meta.insert(0, 'Cell', sc_cell_rename)
        generate_sc_meta['Batch'] = 'sc'
        generate_sc_meta_new = generate_sc_meta.drop(['Cell_type'], axis=1)
        st_cell_rename = [f'ST_{i}' for i in range(1, input_st_data_new.shape[1] + 1)]
        input_st_data.columns = input_st_data_new.columns = st_cell_rename
        input_st_meta = input_st_meta.drop(['Cell'], axis=1)
        input_st_meta.insert(0, 'Cell', st_cell_rename)
        input_st_meta_new = pd.DataFrame({'Cell': st_cell_rename, 'Batch': 'st'})

        all_data = generate_sc_data_new.join(input_st_data_new)
        all_meta = pd.concat([generate_sc_meta_new, input_st_meta_new], ignore_index=True)
        joint_data = joint_analysis(all_data, all_meta['Batch'], ref_batch="st")
        joint_data[joint_data < 0] = 0
        sc_data_new = joint_data.iloc[:, 0:generate_sc_data_new.shape[1]]
        st_data_new = joint_data.iloc[:, generate_sc_data_new.shape[1]:all_data.shape[1]]

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
        os.makedirs(map_save_dir, exist_ok=True)
        meta_dir = os.path.join(map_save_dir, f'meta_{map_save_name}.csv')
        data_dir = os.path.join(map_save_dir, f'data_{map_save_name}.csv')
        st_meta_pred.to_csv(os.path.join(meta_dir))
        st_data_pred.to_csv(os.path.join(data_dir))
        print(f'saving to {meta_dir} and {data_dir}')
        return st_meta_pred, st_data_pred

    def __get_model_input(self, data, cell_target_num):
        # input：data， celltype， bulk & output: label, dic, single_cell
        single_cell = data["input_sc_data"].values.T  # single cell data (600 * 6588)
        index_2_gene = (data["input_sc_data"].index).tolist()
        breed = data["input_sc_meta"]['Cell_type']
        breed_np = breed.values
        breed_set = set(breed_np)
        breed_2_list = list(breed_set)
        dic = {}  # breed_set to index {'B cell': 0, 'Monocyte': 1, 'Dendritic cell': 2, 'T cell': 3}
        label = []  # the label of cell (with index correspond)
        nclass = len(breed_set)

        ntrain = single_cell.shape[0]
        # FeaSize = single_cell.shape[1]
        feature_size = single_cell.shape[1]
        assert nclass == len(cell_target_num.keys()), "cell type num no match!!!"

        for i in range(len(breed_set)):
            dic[breed_2_list[i]] = i
        cell = data["input_sc_meta"]["Cell"].values

        for i in range(cell.shape[0]):
            label.append(dic[breed_np[i]])

        label = np.array(label)

        # label index the data size of corresponding target
        cell_number_target_num = {}
        for k, v in cell_target_num.items():
            cell_number_target_num[dic[k]] = v

        return single_cell, label, breed_2_list, index_2_gene, cell_number_target_num, nclass, ntrain, feature_size

    def __save_generation(self, generate_sc_meta, generate_sc_data, generate_save_dir,
                          generate_save_name, ):
        # saving.....
        if not os.path.exists(generate_save_dir):
            os.makedirs(generate_save_dir)
        path_label_generate_csv = os.path.join(generate_save_dir, f"{generate_save_name}_sc_celltype.csv")
        path_cell_generate_csv = os.path.join(generate_save_dir, f"{generate_save_name}_sc_data.csv")

        generate_sc_meta.to_csv(path_label_generate_csv)
        generate_sc_data.to_csv(path_cell_generate_csv)
        print(f"saving to {path_label_generate_csv} and {path_cell_generate_csv}.")
