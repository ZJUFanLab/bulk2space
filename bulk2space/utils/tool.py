# -*- coding: utf-8 -*-
import sys

from utils.config import project_root

sys.path.insert(0, '/Users/darcy/code/tmp/import_test')

from tqdm import tqdm
from torch import nn
from pytorch_transformers import AdamW
from model.VAE import VAE, BetaVAE_H
from torch.utils.data import Dataset, DataLoader
import torch
import os
import os.path as osp
import pandas as pd
import numpy as np
import copy
import random
import pickle
from collections import defaultdict
from deepforest import CascadeForestClassifier
from torch.utils.tensorboard import SummaryWriter
import scanpy
from scipy.optimize import nnls
from scipy.linalg import solve
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# bulk deconvolution
class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, single_cell, label):
        # 1. Initialize file path or list of file names.
        self.sc = single_cell
        self.label = label

    def __getitem__(self, idx):
        # TODO
        tmp_x = self.sc[idx]
        tmp_y_tag = self.label[idx]

        return (tmp_x, tmp_y_tag)  # tag 分类

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.label.shape[0]


class labelDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, label):
        # 1. Initialize file path or list of file names.
        self.label = label

    def __getitem__(self, idx):
        tmp_y_tag = self.label[idx]

        return tmp_y_tag  # tag 分类

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.label.shape[0]


def train_vae(args, single_cell, cfg, label):

    batch_size = args.batch_size
    feature_size = args.feature_size
    epoch_num = args.epoch_num
    lr = args.learning_rate
    # random_seed = args.random_seed

    if int(args.hidden_lay) == 0:
        hidden_list = [2048, 1024, 512]
    elif int(args.hidden_lay) == 1:
        hidden_list = [4096, 2048, 1024, 512]
    elif int(args.hidden_lay) == 2:
        hidden_list = [8192, 4096, 2048, 1024]
    else:
        print("error define hidden layers")
        exit()
    mid_hidden_size = args.hidden_size
    weight_decay = 5e-4
    dataloader = DataLoader(myDataset(single_cell=single_cell, label=label), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    criterion = nn.MSELoss()
    if args.BetaVAE_H:
        beta = 4
        beta1 = 0.9
        beta2 = 0.999
        # vae = BetaVAE_H(feature_size, hidden_list, mid_hidden_size).cuda()
        vae = VAE(feature_size, hidden_list, mid_hidden_size).cuda()
        optimizer = AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    else:
        vae = VAE(feature_size, hidden_list, mid_hidden_size).cuda()
        optimizer = AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)
        
    
    
    pbar = tqdm(range(epoch_num))
    # writer = SummaryWriter()
    min_loss = 1000000000000000
    vae.train()
    early_stop = 0

    for epoch in pbar:
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            cell_feature, label = data
            cell_feature = torch.tensor(cell_feature, dtype=torch.float32).cuda()
            if args.BetaVAE_H:
                x_recon, total_kld = vae(cell_feature)

                recon_loss = criterion(x_recon, cell_feature)
                beta_vae_loss = recon_loss + beta*total_kld
                loss = beta_vae_loss
            else:
                x_hat, kl_div = vae(cell_feature)
                loss = criterion(x_hat, cell_feature)

                if args.kl_loss:
                    if kl_div is not None:
                        loss += kl_div

                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # writer.add_scalar('loss', loss.item(), epoch)


        if train_loss < min_loss:
            min_loss = train_loss
            # pbar.write("Epoch {}, min loss update to: {:.4f}".format(epoch, train_loss))
            best_vae = copy.deepcopy(vae)
            early_stop = 0
            epoch_final = epoch
        else:
            early_stop += 1
        pbar.set_description('Train Epoch: {}'.format(epoch))
        pbar.set_postfix(loss=f"{train_loss:.4f}", min_loss=f"{min_loss:.4f}")
        if early_stop > args.early_stop and not args.not_early_stop: # use early_stop
            break

    name = "vae"
    # kl_loss BetaVAE_H
    if args.BetaVAE_H:
        name = "BetaVAE"

    path_save = os.path.join(args.save, args.project_name, args.model_choice_1, f"{args.project_name}_{name}_epoch_{epoch_final}_lr_{args.learning_rate}_loss_{train_loss:.2f}.pth")
    if not osp.exists(os.path.join(args.save, args.project_name, args.model_choice_1)):
        os.makedirs(os.path.join(args.save, args.project_name, args.model_choice_1))
    torch.save(best_vae.state_dict(), path_save)
    # writer.close()
    print(f"min loss = {min_loss}")
    return vae


def load_vae(args, cfg):
    assert args.load_path_1 != osp.join(project_root, args.save, args.model_choice_1, ''), "load path has to be assigned!!"
    load_path_1 = args.load_path_1
    # else:
    #     load_path = cfg.load_path_vae

    feature_size = cfg.FeaSize
    assert args.hidden_lay in [0, 1, 2], "error define hidden layers!!"
    if int(args.hidden_lay) == 0:
        hidden_list = [2048, 1024, 512]
    elif int(args.hidden_lay) == 1:
        hidden_list = [4096, 2048, 1024, 512]
    elif int(args.hidden_lay) == 2:
        hidden_list = [8192, 4096, 2048, 1024]
    mid_hidden_size = args.hidden_size
    vae = VAE(feature_size, hidden_list, mid_hidden_size).cuda()

    vae.load_state_dict(torch.load(osp.join(args.save, load_path_1)))
    return vae


def generate_vae(net, args, ratio, single_cell, cfg, label, breed_2_list, index_2_gene, cell_number_target_num=None):
    # pdb.set_trace()
    # net in cuda now
    for p in net.parameters():  # reset requires_grad
        p.requires_grad = False  # avoid computation
    
    net.eval()
    net.cuda()
    cell_all_generate = []
    label_all_generate = []

    all_to_generate = 0
    for x in cell_number_target_num.values():
        all_to_generate += x

    if cell_number_target_num != None:
        epochs = 10000 # 10000次
        ratio = 1
    else:
        epochs = 1
 
    fmt = '{}'.format
    cell_feature = torch.from_numpy(single_cell).float()
    label = torch.from_numpy(label)
    ##############
    
    with torch.no_grad():
        with tqdm(total=all_to_generate) as pbar:
            for epoch in range(epochs):
                key_list = []  # list
                generate_num = 0

                label_list = label.tolist()
                for i in range(len(label_list)):
                    if cell_number_target_num[label_list[i]] <= 0:
                        continue
                    else:
                        cell_number_target_num[label_list[i]] -= 1
                        generate_num += 1
                        key_list.append(i) 
                

                if cell_number_target_num == None or all_to_generate == 0 or len(key_list) == 0:
                    assert all_to_generate == 0 and len(key_list) == 0 
                    break

                # 随机打乱
                random.shuffle(key_list)
                
                label = label.index_select(0, torch.tensor(key_list))
                cell_feature = cell_feature.index_select(0, torch.tensor(key_list))

                dataloader = DataLoader(myDataset(single_cell=cell_feature, label=label), batch_size=300, shuffle=False, pin_memory=True, num_workers=12)
                for batch_idx, data in enumerate(dataloader):  # 一个batch
                    cell_feature_batch, label_batch = data
                    cell_feature_batch = cell_feature_batch.cuda()
                    pbar.set_description('Generate Epoch: {}'.format(epoch))
                    ##############

                    # pbar.set_postfix(remain_to_generate=fmt(all_to_generate))
                    label_batch = label_batch.cpu().numpy()

                    
                    for j in range(ratio):  # 翻倍多少
                        ans_l, _ = net(cell_feature_batch)
                        ans_l = ans_l.cpu().data.numpy()
                        # for i in range(ans_l.shape[0]):
                        cell_all_generate.extend(ans_l)
                        label_all_generate.extend(label_batch)
                
                all_to_generate -= generate_num
                pbar.update(generate_num)

    print("generated done!")
    print("begin data to spatial mapping...")
    generate_sc_meta, generate_sc_data = prepare_data(args, cell_all_generate, label_all_generate, breed_2_list, index_2_gene)
    print("Data have been prepared...")
    return generate_sc_meta, generate_sc_data


def prepare_data(args, cell_all_generate, label_all_generate, breed_2_list, index_2_gene):
    cell_all_generate = np.array(cell_all_generate)
    label_all_generate = np.array(label_all_generate)

    cell_all_generate_csv = pd.DataFrame(cell_all_generate)
    label_all_generate_csv = pd.DataFrame(label_all_generate)

    ids = label_all_generate_csv[0].tolist()
    breeds = []
    for id in ids:
        breeds.append(breed_2_list[id])
    name = ["C_" + str(i + 1) for i in range(label_all_generate.shape[0])]

    label_all_generate_csv.insert(1, "Cell_type", np.array(breeds))
    label_all_generate_csv.insert(1, "Cell", np.array(name))
    label_all_generate_csv = label_all_generate_csv.drop([0], axis=1)

    cell_all_generate_csv = cell_all_generate_csv.T
    cell_all_generate_csv.columns = name
    cell_all_generate_csv.index = index_2_gene

    return label_all_generate_csv, cell_all_generate_csv


# space mapping
def filepath(output, project, dataname, mul):
    # model_dir = osp.join('data/step2/predata', dataset)
    model_dir = osp.join(output, project, 'predata')
    os.makedirs(model_dir, exist_ok=True)
    path = osp.join(model_dir, f'{dataname}_{mul}.pkl')
    return path


def create_st(generate_sc_data, generate_sc_meta, spot_num, cell_num, gene_num, marker_used):
    sc = generate_sc_data
    sc_ct = generate_sc_meta
    cell_name = sorted(list(set(sc_ct.Cell)))

    last_cell_pool = []
    spots = pd.DataFrame()
    meta = pd.DataFrame(columns=['Cell', 'Celltype', 'Spot'])
    sc_ct.index = sc_ct['Cell']
    for i in range(spot_num):
        cell_pool = random.sample(cell_name, cell_num)
        while set(cell_pool) == set(last_cell_pool):
            cell_pool = random.sample(cell_name, cell_num)
        last_cell_pool = cell_pool
        syn_spot = sc[cell_pool].sum(axis=1)
        if syn_spot.sum() > 25000:
            syn_spot *= 20000 / syn_spot.sum()
        spot_name = f'spot_{i + 1}'
        spots.insert(len(spots.columns), spot_name, syn_spot)

        for cell in cell_pool:
            celltype = sc_ct.loc[cell, 'Cell_type']
            row = {'Cell': cell, 'Celltype': celltype, 'Spot': spot_name}
            meta = meta.append(row, ignore_index=True)

    # if hvg_used:
    #     adata = scanpy.AnnData(sc.T)
    #     scanpy.pp.highly_variable_genes(adata, n_top_genes=gene_num)
    #     sc = adata[:, adata.var.highly_variable]
    #     sc = sc.to_df().T
    #     gene_list = sc._stat_axis.values.tolist()
    #     spots = spots.loc[gene_list, :]

    if marker_used:
        adata = scanpy.AnnData(sc.T)

        adata.obs = sc_ct[['Cell_type']]
        scanpy.tl.rank_genes_groups(adata, 'Cell_type', method='wilcoxon')
        marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(gene_num)
        marker_array = np.array(marker_df)
        marker_array = np.ravel(marker_array)
        marker_array = np.unique(marker_array)
        marker = list(marker_array)
        sc = sc.loc[marker, :]
        spots = spots.loc[marker, :]


    # file_path = 'data/step1/simulated'
    # if not osp.exists(file_path):
    #     os.makedirs(file_path)
    # spots.to_csv(os.path.join(file_path, f'data_{dataset}_{dataname}_simulated_st.csv'))
    # meta.to_csv(os.path.join(file_path, f'meta_{dataset}_{dataname}.csv'))
    return sc, sc_ct, spots, meta


def create_sample(sc, st, meta, multiple):
    cell_name = meta.Cell.values.tolist()
    spot_name = meta.Spot.values.tolist()

    # get wrong spot name for negative data
    all_spot = list(set(meta.Spot))
    wrong_spot_name = []
    for sn in spot_name:
        last_spot = all_spot.copy()  # --
        last_spot.remove(sn)  # --
        mul_wrong = random.sample(last_spot, multiple)
        wrong_spot_name.extend(mul_wrong)

    cfeat_p_list, cfeat_n_list = [], []

    for c in cell_name:
        cell_feat = sc[c].values.tolist()

        cfeat_p_list.append(cell_feat)
        cfeat_m = [cell_feat * multiple]
        cfeat_n_list.extend(cfeat_m)

    cfeat_p = np.array(cfeat_p_list)  # [n, d]
    cfeat_n = np.array(cfeat_n_list).reshape(-1, cfeat_p.shape[1])  # [n*m, d]

    # positive spot features
    sfeat_p_list = []
    for s in spot_name:
        spot_feat = st[s].values.tolist()
        sfeat_p_list.append(spot_feat)
        # sfeat_p = np.vstack((sfeat_p, spot_feat))
    sfeat_p = np.array(sfeat_p_list)  # [n, d]

    mfeat_p = sfeat_p - cfeat_p
    feat_p = np.hstack((cfeat_p, sfeat_p))
    feat_p = np.hstack((feat_p, mfeat_p))
    print('sucessfully create positive data')

    # negative spot features
    sfeat_n_list = []
    for s in wrong_spot_name:
        spot_feat = st[s].values.tolist()
        sfeat_n = sfeat_n_list.append(spot_feat)
    sfeat_n = np.array(sfeat_n_list)

    mfeat_n = sfeat_n - cfeat_n
    feat_n = np.hstack((cfeat_n, sfeat_n))
    feat_n = np.hstack((feat_n, mfeat_n))
    print('sucessfully create negative data')

    return feat_p, feat_n


def get_data(pos, neg):
    X = np.vstack((pos, neg))
    y = np.concatenate((np.ones(pos.shape[0]), np.zeros(neg.shape[0])))
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)

    return X, y


def save(data, output, project, dataname, mul):
    # model_dir = osp.join('data/predata', dataset)
    model_dir = osp.join(output, project, 'predata')
    os.makedirs(model_dir, exist_ok=True)
    data_output = open(osp.join(model_dir, f'{dataname}_{mul}.pkl'), 'wb')
    pickle.dump(data, data_output)
    data_output.close()
    print(f'save {dataname} ok！')


def load(output, project, dataname, mul, np=False):
    # model_dir = osp.join('data/predata', dataset)
    model_dir = osp.join(output, project, 'predata')
    data_input = open(osp.join(model_dir, f'{dataname}_{mul}.pkl'), 'rb')
    data = pickle.load(data_input)
    data_input.close()
    print(f'load {dataname} ok!')
    if np:
        return data.numpy()
    return data


class CreatData:
    def __init__(self, generate_sc_data, generate_sc_meta, st_data, args):
        self.args = args
        self.generate_sc_data = generate_sc_data
        self.generate_sc_meta = generate_sc_meta
        self.st_data = st_data

        sc_gene = self.generate_sc_data._stat_axis.values.tolist()
        st_gene = self.st_data._stat_axis.values.tolist()
        intersect_gene = list(set(sc_gene).intersection(set(st_gene)))
        self.generate_sc_data = self.generate_sc_data.loc[intersect_gene]
        self.st_data = self.st_data.loc[intersect_gene]


    def cre_data(self):
        ftrain = filepath(self.args.output_path, self.args.
                          project_name, self.args.xtrain, self.args.mul_train)

        if not osp.isfile(ftrain) and self.args.train_model_2:
            # train
            print('preparing train data...')
            sc_train, _, st_train, meta_train = create_st(self.generate_sc_data, self.generate_sc_meta, self.args.spot_num, self.args.cell_num,
                                                          self.args.top_marker_num, self.args.marker_used)
            pos_train, neg_train = create_sample(sc_train, st_train, meta_train, self.args.mul_train)
            xtrain, ytrain = get_data(pos_train, neg_train)
            save(xtrain, self.args.output_path, self.args.project_name, self.args.xtrain, self.args.mul_train)
            save(ytrain, self.args.output_path, self.args.project_name, self.args.ytrain, self.args.mul_train)


        if osp.isfile(ftrain) and self.args.train_model_2:
            print('train data already prepared.')


class Runner:
    def __init__(self, generate_sc_data, generate_sc_meta, st_data, st_meta, args):
        self.args = args
        # self.log_dir = get_dump_path(args)
        # self.model_dir = os.path.join(self.args.output_path, 'save_model')
        # self.output_dir = os.path.join(self.args.output_path, 'output_data')
        self.model_dir = self.args.save
        self.output_dir = self.args.output_path
        self.projrct = self.args.project_name

        # self.dataset = self.args.dataset
        self.writer = SummaryWriter()
        self.train_model_2 = args.train_model_2

        self.sc_test_allgene = generate_sc_data
        self.cell_type = generate_sc_meta
        self.st_data = st_data
        self.meta_test = st_meta

        # load data
        if self.train_model_2:
            # self.xtrain = load(self.dataset, self.args.xtrain, self.args.mul_train, np=True)  # 5890, 11787])
            # self.ytrain = load(self.dataset, self.args.ytrain, self.args.mul_train, np=True)  # [5890])
            self.xtrain = load(self.output_dir, self.projrct, self.args.xtrain, self.args.mul_train, np=True)  # 5890, 11787])
            self.ytrain = load(self.output_dir, self.projrct, self.args.ytrain, self.args.mul_train, np=True)  # [5890])

        # self.sc_test_allgene = load_sc_data(args.sc_path, 'data', args.dataset, args.dataname)
        # self.sc_test, self.st_test = load_st_data(args.st_path, args.sc_path, args.dataset, args.dataname,
        #                                           args.gene_num, args.st_data, args.marker_used)

        self.sc_test = self.sc_test_allgene
        self.st_test = self.st_data
        sc_gene = self.sc_test._stat_axis.values.tolist()
        st_gene = self.st_test._stat_axis.values.tolist()
        intersect_gene = list(set(sc_gene).intersection(set(st_gene)))
        self.sc_test = self.sc_test.loc[intersect_gene]
        self.st_test = self.st_test.loc[intersect_gene]

        if self.args.marker_used:
            print('select top %d marker genes of each cell type...' %self.args.top_marker_num)

            # sc = scanpy.AnnData(self.sc_test.T)
            # scanpy.pp.highly_variable_genes(sc, n_top_genes=self.args.highly_variable_gene_num)
            # self.sc_test = sc[:, sc.var.highly_variable]
            # self.sc_test = self.sc_test.to_df().T
            # gene_list = self.sc_test._stat_axis.values.tolist()
            # self.st_test = self.st_test.loc[gene_list, :]

            sc = scanpy.AnnData(self.sc_test.T)
            sc.obs = self.cell_type[['Cell_type']]
            scanpy.tl.rank_genes_groups(sc, 'Cell_type', method='wilcoxon')
            marker_df = pd.DataFrame(sc.uns['rank_genes_groups']['names']).head(args.top_marker_num)
            marker_array = np.array(marker_df)
            marker_array = np.ravel(marker_array)
            marker_array = np.unique(marker_array)
            marker = list(marker_array)
            self.sc_test = self.sc_test.loc[marker, :]
            self.st_test = self.st_test.loc[marker, :]


        # self.meta_test = load_st_meta(args.st_path, args.st_meta)

        if self.args.model_choice_2 == 'df':
            self.model = CascadeForestClassifier(random_state=self.args.random_seed, n_jobs=os.cpu_count() // 4 * 3,
                                                 verbose=0)

        # self.cell_type = load_sc_data(args.sc_path, 'celltype', self.dataset, args.dataname)
        # pdb.set_trace()

        breed = self.cell_type['Cell_type']
        breed_np = breed.values
        breed_set = set(breed_np)
        self.id2label = sorted(list(breed_set))
        self.label2id = {label: idx for idx, label in enumerate(self.id2label)}
        self.cell2label = dict()
        self.label2cell = defaultdict(set)
        for row in self.cell_type.itertuples():
            cell_name = getattr(row, 'Cell')
            cell_type = self.label2id[getattr(row, 'Cell_type')]
            self.cell2label[cell_name] = cell_type
            self.label2cell[cell_type].add(cell_name)

    def run(self):
        if self.train_model_2:
            self.model.fit(self.xtrain, self.ytrain)
            self._save_model(os.path.join(self.model_dir, self.projrct, self.args.model_choice_2))

            print('model trained sucessfully, start saving output ...')
            self.cre_csv()

        if not self.train_model_2:
            print('model has been loaded ...')
            self.cre_csv()

    def cre_csv(self):
        if not self.train_model_2:
            self._load_model(self.args.load_path_2)

        # data
        cell_name = self.sc_test.columns.values.tolist()
        spot_name = self.st_test.columns.tolist()  # list of spot name ['spot_1', 'spot_2', ...]
        cfeat = self.sc_test.values.T
        cell_num = len(cell_name)
        spot_num = len(spot_name)
        if self.args.max_cell_in_diff_spot_ratio is None:
            max_cell_in_diff_spot = None
        else:
            max_cell_in_diff_spot = int(self.args.max_cell_in_diff_spot_ratio * self.args.k * spot_num / cell_num)

        def joint_predict(ratio):
            score_triple_list = list()
            spot2cell = defaultdict(set)
            cell2spot = defaultdict(set)
            spot2ratio = dict()

            from multiprocessing.pool import Pool
            re_list = []
            process_pool = Pool(8)

            print('Calculating scores...')
            for spot_indx in range(len(spot_name)):
                spot = spot_name[spot_indx]  # spotname
                re_list.append(process_pool.apply_async(predict_for_one_spot, (
                self.model, self.st_test, cell_name, spot_name, spot_indx, cfeat)))

            process_pool.close()
            process_pool.join()
            print('Calculating scores done.')

            for r in re_list:
                spot_indx, predict = r.get()
                spot = spot_name[spot_indx]  # spotname
                for c, p in zip(cell_name, predict):
                    score_triple_list.append((c, spot, p))  # (cell, spot, score)
                spot2ratio[spot] = np.round(ratio[spot_indx] * self.args.k)  # [n1, n2, ...]

            score_triple_list = sorted(score_triple_list, key=lambda x: x[2], reverse=True)
            for c, spt, score in score_triple_list:
                # cell name, spot name, score
                if max_cell_in_diff_spot is not None and len(cell2spot[c]) == max_cell_in_diff_spot:
                    continue
                if len(spot2cell[spt]) == self.args.k:
                    continue
                cell_class = self.cell2label.get(c)
                if cell_class is None:
                    continue
                if spot2ratio[spt][cell_class] > 0:
                    spot2ratio[spt][cell_class] -= 1
                    spot2cell[spt].add(c)
                    cell2spot[c].add(spt)
                else:
                    continue

            cell_list, spot_list, spot_len = list(), list(), list()
            df_spots = pd.DataFrame()

            order_list = spot_name
            for spot in order_list:
                if spot2cell.get(spot):
                    cells = spot2cell.get(spot)
                    cell_num = len(cells)
                    cell_list.extend(sorted(list(cells)))
                    spot_list.extend([spot] * cell_num)
                    spot_len.append(cell_num)
                    cell_pool = list(cells)
                    cell_pool.sort()

                    predict_spot = self.sc_test_allgene[cell_pool]
                    df_spots = pd.concat([df_spots, predict_spot], axis=1)
            return cell_list, spot_list, spot_len, df_spots

        ratio = self.__calc_ratio()  # [spot_num, class_num]

        cell_list, spot_list, spot_len, df_spots = joint_predict(ratio)
        meta = {'Cell': cell_list, 'Spot': spot_list}
        df = pd.DataFrame(meta)
        # pdb.set_trace()
        self.cell_type = self.cell_type.reset_index(drop=True)
        df_meta = pd.merge(df, self.cell_type, how='left')
        df_meta = df_meta[['Cell', 'Cell_type', 'Spot']]
        df_meta = pd.merge(df_meta, self.meta_test, how='inner')

        df_meta = df_meta.rename(columns={'xcoord': 'Spot_xcoord', 'ycoord': 'Spot_ycoord'})

        coord = self.meta_test[['xcoord', 'ycoord']].to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coord)
        distances, indices = nbrs.kneighbors(coord)
        radius = distances[:, -1] / 2
        radius = radius.tolist()
        all_coord = df_meta[['Spot_xcoord', 'Spot_ycoord']].to_numpy()
        all_radius = list()

        for i in range(len(spot_len)):
            a = [radius[i]] * spot_len[i]
            all_radius.extend(a)

        length = np.random.uniform(0, all_radius)
        angle = np.pi * np.random.uniform(0, 2, all_coord.shape[0])

        x = all_coord[:, 0] + length * np.cos(angle)
        y = all_coord[:, 1] + length * np.sin(angle)
        cell_coord = {'Cell_xcoord': np.around(x, 2).tolist(), 'Cell_ycoord': np.around(y, 2).tolist()}
        df_cc = pd.DataFrame(cell_coord)
        df_meta = pd.concat([df_meta, df_cc], axis=1)

        cell_rename = [f'C_{i}' for i in range(1, df_spots.shape[1] + 1)]
        df_spots.columns = cell_rename
        df_meta = df_meta.drop(['Cell'], axis=1)
        df_meta.insert(0, "Cell", cell_rename)

        #  save df
        os.makedirs(self.output_dir, exist_ok=True)

        df_meta.to_csv(os.path.join(self.output_dir, self.args.project_name, f'meta_{self.args.project_name}_{self.args.k}.csv'))
        df_spots.to_csv(os.path.join(self.output_dir, self.args.project_name, f'data_{self.args.project_name}_{self.args.k}.csv'))
        print('save csv ok')

    def __calc_ratio(self):

        # pdb.set_trace()
        label_devide_data = dict()
        for label, cells in self.label2cell.items():
            label_devide_data[label] = self.sc_test[list(cells)]

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

        num_spot = self.st_test.values.shape[1]

        spot_ratio_values = np.zeros((num_spot, max_decade))  # (spot_num, label_num)
        spot_values = self.st_test.values  # (gene_num, spot_num)

        # pdb.set_trace()
        for i in range(num_spot):
            ratio_list = [0 for x in range(max_decade)]
            spot_rep = spot_values[:, i].reshape(-1, 1)
            spot_rep = spot_rep.reshape(spot_rep.shape[0], )

            ratio = nnls(single_cell_matrix, spot_rep)[0]

            ratio_list = [r for r in ratio]
            ratio_list = (ratio_list / np.sum([ratio_list], axis=1)[0]).tolist()

            for j in range(max_decade):
                spot_ratio_values[i, j] = ratio_list[j]
        return spot_ratio_values


    def _load_model(self, save_path):
        # TODO: path
        # if args.model_type == 'df':
        #     # save_path = osp.join(save_path, 'dir')
        #     self.model.load(save_path)
        # assert args.load_path_2 != osp.join(cfg.project_root, args.save, args.model_choice_2, ''), "load path has to be assigned!!"
        self.model.load(save_path)
        print(f"loading model done!")

    def _save_model(self, save_path):
        # TODO: path
        # if args.model_type == 'df':
        #     self.model.save(save_path)

        self.model.save(save_path)
            # pass
        print(f"saving model done!")
        return save_path


def creat_pre_data(st, cell_name, spot_name, spot_indx, cfeat, return_np=False):
    spot = spot_name[spot_indx]
    spot_feat = st[spot].values
    tlist = np.isnan(spot_feat).tolist()
    tlist = [i for i, x in enumerate(tlist) if x == True]
    assert len(tlist) == 0

    sfeat = np.tile(spot_feat, (len(cell_name), 1))
    mfeat = sfeat - cfeat
    feat = np.hstack((cfeat, sfeat))
    feat = np.hstack((feat, mfeat))
    if not return_np:
        feat = torch.from_numpy(feat).type(torch.FloatTensor)

    tlist = np.isnan(sfeat).tolist()
    tlist = [i for i, x in enumerate(tlist) if x == True]
    assert len(tlist) == 0

    tlist = np.isnan(cfeat).tolist()
    tlist = [i for i, x in enumerate(tlist) if x == True]
    assert len(tlist) == 0

    return feat


def predict_for_one_spot(model, st_test, cell_name, spot_name, spot_indx, cfeat):
    feats = creat_pre_data(st_test, cell_name, spot_name, spot_indx, cfeat, return_np=True)
    outputs = model.predict_proba(feats)[:, 1]
    # outputs = np.where(outputs>0.5, 1, 0)
    predict = outputs.tolist()
    return spot_indx, predict


def aprior(gamma_hat, axis=None):
    m = np.mean(gamma_hat, axis=axis)
    s2 = np.var(gamma_hat, ddof=1, axis=axis)
    return (2 * s2 + np.power(m, 2)) / s2


def bprior(gamma_hat, axis=None):
    m = np.mean(gamma_hat, axis=axis)
    s2 = np.var(gamma_hat, ddof=1, axis=axis)
    return (m * s2 + np.power(m, 3)) / s2


def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2 + a - 1)


def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = np.sum(~np.isnan(sdat), axis=1)
    g_old = g_hat
    d_old = d_hat
    change = 1
    count = 0
    while change > conv:
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = np.nansum(np.power(sdat - np.dot(np.expand_dims(g_new, axis=1), np.ones((1, sdat.shape[1]))), 2), axis=1)
        d_new = postvar(sum2, n, a, b)
        change = np.max((np.max(np.abs(g_new - g_old) / g_old), np.max(np.abs(d_new - d_old) / d_old)))
        g_old = g_new
        d_old = d_new
        count += 1
    return np.concatenate((np.expand_dims(g_new, axis=1), np.expand_dims(d_new, axis=1)), axis=1)


def joint_analysis(dat, batch, mod=None, par_prior=True, proir_plots=False, mean_only=False, ref_batch=None):
    rownames = dat.index
    colnames = dat.columns
    dat = np.array(dat)
    batch_levels = batch.drop_duplicates()

    batches = []
    ref_index = 0
    zero_rows_list = []
    for i, batch_level in enumerate(batch_levels):
        idx = batch.isin([batch_level])
        if batch_level == ref_batch:
            ref_index = i
        batches.append(idx.reset_index().loc[lambda d: d.Batch == True].index)
        batch_dat = dat[:, idx]
        for row in range(np.size(batch_dat, 0)):
            if np.var(batch_dat[row, :], ddof=1) == 0:
                zero_rows_list.append(row)
    zero_rows = list(set(zero_rows_list))
    keep_rows = list(set(range(dat.shape[0])).difference(set(zero_rows_list)))
    dat_origin = dat
    dat = dat[keep_rows, :]
    batchmod = pd.get_dummies(batch, drop_first=False, prefix='batch')
    batchmod['batch_' + ref_batch] = 1
    ref = batchmod.columns.get_loc('batch_' + ref_batch)
    design = np.array(batchmod)
    n_batch = batch_levels.shape[0]
    n_batches = [len(x) for x in batches]

    B_hat = solve(np.dot(design.T, design), np.dot(design.T, dat.T))
    grand_mean = np.expand_dims(B_hat[ref, :], axis=1)
    ref_dat = dat[:, batches[ref_index]]
    var_pool = np.expand_dims(np.dot(np.square(ref_dat - np.dot(design[batches[ref_index], :], \
                                                                B_hat).T),
                                     np.ones(n_batches[ref_index]).T * 1 / n_batches[ref_index]), axis=1)
    stand_mean = np.dot(grand_mean, np.ones((1, batch.shape[0])))
    s_data = (dat - stand_mean) / np.dot(np.sqrt(var_pool), np.ones((1, batch.shape[0])))
    batch_design = design
    gamma_hat = solve(np.dot(batch_design.T, batch_design), np.dot(batch_design.T, s_data.T))
    delta_hat = np.empty([0, s_data.shape[0]])
    for i in batches:
        row_vars = np.expand_dims(np.nanvar(s_data[:, i], axis=1, ddof=1), axis=0)
        delta_hat = np.concatenate((delta_hat, row_vars), axis=0)
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    a_prior = aprior(delta_hat, axis=1)
    b_prior = bprior(delta_hat, axis=1)
    results = []
    gamma_star = np.empty((n_batch, s_data.shape[0]))
    delta_star = np.empty((n_batch, s_data.shape[0]))
    for j, batch_level in enumerate(batch_levels):
        i = batchmod.columns.get_loc('batch_' + batch_level)
        results.append(it_sol(s_data[:, batches[j]], gamma_hat[i, :], delta_hat[j, :], gamma_bar[i], t2[i], a_prior[j],
                              b_prior[j]).T)
    for j, batch_level in enumerate(batch_levels):
        gamma_star[j, :] = results[j][0]
        delta_star[j, :] = results[j][1]
    gamma_star[ref_index, :] = 0
    delta_star[ref_index, :] = 1
    bayesdata = s_data
    for i, batch_index in enumerate(batches):
        bayesdata[:, batch_index] = (bayesdata[:, batch_index] - np.dot(batch_design[batch_index, :], gamma_star).T) / \
                                    np.dot(np.sqrt(np.expand_dims(delta_star[i], axis=1)), np.ones((1, n_batches[i])))

    bayesdata = (bayesdata * np.dot(np.sqrt(var_pool), np.ones((1, dat.shape[1])))) + stand_mean
    bayesdata[:, batches[ref_index]] = dat[:, batches[ref_index]]
    if len(zero_rows) > 0:
        dat_origin[keep_rows, :] = bayesdata
        bayesdata = pd.DataFrame(dat_origin, index=rownames, columns=colnames)
    bayesdata[bayesdata < 0] = 0
    return bayesdata


def knn(data, query, k):
    tree = KDTree(data)
    dist, ind = tree.query(query, k)
    return dist, ind
