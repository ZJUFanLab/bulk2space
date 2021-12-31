# -*- coding: utf-8 -*-
import os.path as osp
from easydict import EasyDict as edict
import argparse


# 经常需要变动的参数放args，比较固定的放cfg。

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def loadArgums(cfg):
    parser = argparse.ArgumentParser(description='bulk to spatial')

    parser.add_argument('--project_name', '-pn', default='demo', help='project name', type=str)
    # path
    parser.add_argument('--data_path', '-dp', default='example_data', help='data path')
    parser.add_argument('--save', '-s', default='save_model', help='save model: save_model')
    parser.add_argument('--output_path', '-op', default='output_data', help='save pth to')

    # input data
    parser.add_argument('--input_sc_meta_path', '-iscm', default='demo_sc_meta.csv', help='input sc meta path')
    parser.add_argument('--input_sc_data_path', '-iscd', default='demo_sc_data.csv', help='input sc data path')
    parser.add_argument('--input_bulk_path', '-ib', default='demo_bulk.csv', help='input bulk data')
    parser.add_argument('--input_st_meta_path', '-istm', default='demo_st_meta.csv', help='input st meta path')
    parser.add_argument('--input_st_data_path', '-istd', default='demo_st_data.csv', help='input st data path')


    # bulk deconvolution
    parser.add_argument('--load_path_1', default='', help='path to load vae model')
    parser.add_argument('--load_model_1', '-m1', default=False, help='False--train True--load model', type=str2bool)
    parser.add_argument('--model_choice_1', '-mc1', default="vae", help='vae')
    # parser.add_argument("--highly_variable_gene_num", default=3000, type=int)
    parser.add_argument("--top_marker_num", default=500, type=int)
    parser.add_argument('--hidden_size', '-hs', default=256, help='hidden_size eg:128', type=int)
    parser.add_argument('--random_seed', '-rs', default=12345, help='random seed', type=int)
    parser.add_argument('--learning_rate', '-lr', default=0.0001, help='learning_rate', type=float)
    parser.add_argument('--batch_size', '-bs', default=512, help='batch_size eg:256', type=int)
    parser.add_argument('--early_stop', default=50, type=int)
    parser.add_argument('--feature_size', '-fs', default=6588, help='feature_size eg:6588', type=int)
    parser.add_argument('--hidden_lay', '-hl', default=0,
                        help='VAE lay choice: 0:[2048, 1024, 512] \n 1: [4096, 2048, 1024, 512] \n 2: [8192, 4096, 2048, 1024]',
                        type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--epoch_num', '-ep', default=3500, help='num epochs eg:500000', type=int)
    parser.add_argument('--num_workers', default=12, help='num_workers', type=int)
    parser.add_argument('--ratio_num', default=1, type=float)
    parser.add_argument("--kl_loss", action='store_const', default=False, const=True)
    parser.add_argument("--BetaVAE_H", action='store_const', default=False, const=True)
    parser.add_argument("--not_early_stop", action='store_const', default=False, const=True)

    # space mapping
    parser.add_argument('--spot_data', default=True, help='True--spot st data like 10x/ST False--target gene st data like merfish', type=str2bool)
    parser.add_argument('--xtrain', default='xtrain', type=str)
    parser.add_argument('--ytrain', default='ytrain', type=str)
    parser.add_argument('--xtest', default='xtest', type=str)
    parser.add_argument('--ytest', default='ytest', type=str)
    parser.add_argument('--mul_train', default=1, type=int)
    parser.add_argument('--mul_test', default=5, type=int)
    parser.add_argument('--train_model_2', '-m2', default=True, type=str2bool)
    parser.add_argument('--spot_num', default=500, type=int)
    parser.add_argument('--cell_num', default=10, type=int)
    # parser.add_argument('--hvg_used', default=True, type=bool)
    parser.add_argument('--marker_used', default=True, type=bool)
    parser.add_argument("--model_choice_2", '-mc2', default="df", choices=['mlp', 'svc', 'lr', 'dt', 'rf', 'gbdt', 'df'], type=str)
    parser.add_argument('--max_cell_in_diff_spot_ratio', default=None, type=int)
    parser.add_argument('--load_path_2', default='', help='path to load df model')
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--previous_project_name', '-ppn', default='demo', help='previous project name', type=str)


    # torthlight
    parser.add_argument("--no_tensorboard", default=False, action="store_true")
    parser.add_argument("--dump_path", default="dump", type=str, help="Experiment dump path")

    # path deal
    args = parser.parse_args()
    cfg.FeaSize = args.feature_size

    args.input_sc_meta_path = osp.join(cfg.data_root, args.data_path, args.input_sc_meta_path)
    args.input_sc_data_path = osp.join(cfg.data_root, args.data_path, args.input_sc_data_path)
    args.input_bulk_path = osp.join(cfg.data_root, args.data_path, args.input_bulk_path)
    args.input_st_meta_path = osp.join(cfg.data_root, args.data_path, args.input_st_meta_path)
    args.input_st_data_path = osp.join(cfg.data_root, args.data_path, args.input_st_data_path)
    args.save = osp.join(project_root, args.save)
    args.output_path = osp.join(project_root, args.output_path)
    args.load_path_1 = osp.join(project_root, args.save, args.load_path_1)
    args.load_path_2 = osp.join(project_root, args.save, args.load_path_2)
    args.dump_path = osp.join(project_root, args.dump_path)

    args.exp_name = args.project_name
    args.exp_id = "LR_" + str(args.learning_rate) + "_hiddenSize_" + str(args.hidden_size) + "_lay_choice_" + str(args.hidden_lay)

    return args



# path
this_dir = osp.dirname(__file__)
project_root = osp.abspath(osp.join(this_dir, '..'))
cfg = edict()
cfg.data_root = osp.abspath(osp.join(this_dir, '..', 'data'))


