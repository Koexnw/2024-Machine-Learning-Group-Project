import argparse
from ast import parse
import os
import time
import numpy as np
import copy
import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
from openpyxl import load_workbook
from utils import utils
from models.BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import scipy.sparse as sp
import models.gaussian_diffusion_imp as gd
from models.DNN import DNN
import models.gaussian_diffusion as gd1
import models.Diffreader
# import models.Diffreader
from copy import deepcopy
import multiprocessing
import random
from torch.utils.data import Dataset
from models.general import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from models.reranker import *
import pandas as pd
from models.general.Diffrec import Diffrec
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from models.CAM_AE import CAM_AE
from models.CAM_AE_multihops import CAM_AE_multihops


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        key_columns = ['user_id','item_id','time']
        if 'label' in self.data_df['train'].columns: # Add label for CTR prediction
            key_columns.append('label')
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        if 'label' in key_columns:
            positive_num = (self.all_df.label==1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num, positive_num/self.all_df.shape[0]*100))



class Diffirunner(object):
    def evaluate_method(self,predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        """
        evaluations = dict()
        # sort_idx = (-predictions).argsort(axis=1)
        # gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
        # ↓ As we only have one positive sample, comparing with the first item will be more efficient. 
        if self.printflag==1:
            self.vpredictions=predictions
        if self.printflag==2:
            self.tpredictions=predictions
        gt_rank = (predictions >= predictions[:,0].reshape(-1,1)).sum(axis=-1)
        # if (gt_rank!=1).mean()<=0.05: # maybe all predictions are the same
        # 	predictions_rnd = predictions.copy()
        # 	predictions_rnd[:,1:] += np.random.rand(predictions_rnd.shape[0], predictions_rnd.shape[1]-1)*1e-6
        # 	gt_rank = (predictions_rnd > predictions[:,0].reshape(-1,1)).sum(axis=-1)+1
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        return evaluations
    
    def __init__(self,args):
        self.data_path=args.data_path
        self.lr=args.lr
        self.weight_decay=args.weight_decay
        self.batch_size=args.batch_size
        self.epoch=args.epoch
        
        # self.tst_w_val=args.tst_w_val
        self.cuda=args.cuda
        self.gpu=args.gpu
        self.save_path=args.save_path
        self.log_name=args.log_name
        self.round=args.round
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        # params for the model
        self.time_type=args.time_type
        self.dims=args.dims
        self.norm=args.norm
        self.emb_size=args.emb_size
        self.optimizer_name = args.optimizer
        self.learning_rate=args.lr
        self.weight_decay=args.weight_decay
        self.pin_memory = args.pin_memory
        # params for diffusion
        self.mean_type=args.mean_type
        self.steps=args.steps
        self.w_min=args.w_min
        self.w_max=args.w_max
        self.noise_schedule=args.noise_schedule
        self.noise_scale=args.noise_scale
        self.noise_min=args.noise_min
        self.noise_max=args.noise_max
        self.sampling_noise=args.sampling_noise
        self.sampling_steps=args.sampling_steps
        self.reweight=args.reweight
        self.topN=[int(x) for x in args.topk.split(',')]
        self.num_workers=args.num_workers
        self.model=None
        self.diffusion=None
        self.time = None
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topN[0]) if not len(args.main_metric) else args.main_metric # early stop based on main_metric
        self.main_topk = int(self.main_metric.split("@")[1])
        self.log_path = os.path.dirname(args.log_file) # path to save predictions
        self.save_appendix = args.log_file.split("/")[-1].split(".")[0] # appendix for prediction saving
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.train_models = args.train
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topN[0]) if not len(args.main_metric) else args.main_metric # early stop based on main_metric
        self.model_path=args.model_path
        self.printflag=0
        self.resultflag=0

        corpus = BaseReader(args)
        model = Diffrec(args, corpus).to(args.device)
        self.n_hop=args.n_hop
        self.pre_name=args.pre_name
	# Define dataset
        self.data_dict = dict()
        for phase in ['train', 'dev', 'test']:
            self.data_dict[phase] = Diffrec.Dataset(model, corpus, phase)
            self.data_dict[phase].prepare()
            pass
        self.test_loader=None, 
        self.test_y_data=None, 
        self.train_data=None, 
        self.neg_validlist=None
        self.valid_y_data=None
        self.neg_testlist=None
        self.vpredictions=None
        self.tpredictions=None
        self.imp=args.imp
        self.data_path=args.data_path

        self.test_loader_sec_hop=None


    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=200,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=10,
                            help='The number of epochs when dev results drop continuously.')
 
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='AdamW',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=0,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='5,10,20,50',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='NDCG,HR',
                            help='metrics: NDCG, HR')
        parser.add_argument('--main_metric', type=str, default='',
                            help='Main metric to determine the best model.')
        # parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
        #分割线
        parser.add_argument('--data_path', type=str, default='../data/datasets/', help='load data path')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.0)
        # parser.add_argument('--batch_size', type=int, default=400)
        parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
        # parser.add_argument('--topN', type=str, default='[5, 10, 20, 50]')
        # parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
        parser.add_argument('--cuda', action='store_true', help='use CUDA')
        # parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
        parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
        parser.add_argument('--log_name', type=str, default='log', help='the log name')
        parser.add_argument('--round', type=int, default=1, help='record the experiment')

        # params for the model
        parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
        parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
        parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
        parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
        parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')  #ii
        parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')#ii
        # params for diffusion
        parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
        parser.add_argument('--steps', type=int, default=100, help='diffusion steps')#ii
        parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
        parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')#ii
        parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')#ii
        parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')#ii
        parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
        parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
        parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
        # parser.add_argument('--model_path', type=str, default='',help='Model save path.')
        parser.add_argument('--imp', type=bool, default=0, help='to use improved method or not')
        parser.add_argument('--n_hop', type=int, default=2, help='assign different weight to different timestep or not')
        parser.add_argument('--pre_name', type=str, default='two_hop_rates_items_yelp2018', help='assign different weight to different timestep or not')
        return parser

    class DataDiffusion(Dataset):
        def __init__(self, data):
            self.data = data
        def __getitem__(self, index):
            item = self.data[index]
            return item
        def __len__(self):
            return len(self.data)
    

    def getpredict(self):
        if self.resultflag==0:
            
            self.resultflag=self.resultflag+1
            return self.vpredictions
        else:
            return self.tpredictions

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        logging.info('Optimizer: ' + self.optimizer_name)
        # optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
        #     model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(model.parameters(),
    lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    def evaluate(self,data_loader, data_te, mask_his, topN,excluded_items_per_user):
        # print(len(excluded_items_per_user))
        # print(mask_his.shape[0])
        # print(f'fuyangben {len(excluded_items_per_user[0])}')
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        device = torch.device("cuda:0" if self.cuda else "cpu")
        self.model.eval()  #  在评估函数中也要设置评估模式
        self.diffusion.eval() # diffusion也需要设置评估模式
        e_idxlist = list(range(mask_his.shape[0]))
        e_N = mask_his.shape[0]

        predictions = []
        target_items = []
        for i in range(e_N):
            target_items.append(data_te[i, :].nonzero()[1].tolist())

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, e_N)
                his_data = mask_his[e_idxlist[start_idx:end_idx]]
                batch = batch.to(device)
                prediction = self.diffusion.p_sample(self.model, batch, self.sampling_steps, self.sampling_noise)  # 这里假设你已经定义了 diffusion.p_sample
                # prediction[his_data.nonzero()] = -np.inf

                batch_predictions = prediction.cpu().numpy()

                for user_idx_in_batch in range(len(batch)):
                    global_user_idx = start_idx + user_idx_in_batch

                    excluded_items = excluded_items_per_user[global_user_idx-1]
                    # print(global_user_idx-1)
                    if target_items[global_user_idx]:
                        user_target = target_items[global_user_idx][0]
                        user_predictions = batch_predictions[user_idx_in_batch]

                        all_indices = np.arange(user_predictions.shape[0])
                        exclude_indices = np.setdiff1d(all_indices, excluded_items + [user_target])
                        user_predictions[exclude_indices] = -np.inf

                        # user_prediction_target_first = np.zeros_like(user_predictions)
                        user_prediction_target_first = np.full_like(user_predictions, -np.inf)
                        user_prediction_target_first[0] = user_predictions[user_target]
                        remaining_predictions = user_predictions[np.setdiff1d(all_indices, exclude_indices)]
                        user_prediction_target_first[1:1+len(remaining_predictions)] = remaining_predictions
                        # remaining_predictions = user_predictions[np.setdiff1d(all_indices, exclude_indices+[user_target])]
                        # user_prediction_target_first[1:len(remaining_predictions)] = remaining_predictions
                        predictions.append(user_prediction_target_first)

                    else:
                        # print(f"User {global_user_idx} has no ground truth items. Skipping.")
                        continue

        predictions = np.array(predictions)
        # print(predictions.shape)
        # 假设 finite_counts_per_row 是计算得到的列表
        finite_counts_per_row = np.sum(np.isfinite(predictions), axis=1)
        # print(len(finite_counts_per_row))
        # 检查是否所有元素都相等
        all_equal = np.all(finite_counts_per_row == finite_counts_per_row[0])
        # print(finite_counts_per_row[0])
        # 打印结果
        # if all_equal:
        #     print("All elements in the list are equal.")
        # else:
        #     print("Not all elements in the list are equal.")
        metrics = self.metrics
        
        test_results = self.evaluate_method(predictions, topN, metrics)

        return test_results
    
    def evaluate_imp(self,data_loader,data_loader_sec_hop,data_te,mask_his,topN,excluded_items_per_user):
        # print(len(excluded_items_per_user))
        # print(mask_his.shape[0])
        # print(f'fuyangben {len(excluded_items_per_user[0])}')
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        device = torch.device("cuda:0" if self.cuda else "cpu")
        self.model.eval()  #  在评估函数中也要设置评估模式
        self.diffusion.eval() # diffusion也需要设置评估模式
        e_idxlist = list(range(mask_his.shape[0]))
        e_N = mask_his.shape[0]

        predictions = []
        target_items = []
        for i in range(e_N):
            target_items.append(data_te[i, :].nonzero()[1].tolist())

        with torch.no_grad():
            for (batch_idx, batch), (batch_idx_2, batch_2) in zip(enumerate(data_loader), enumerate(data_loader_sec_hop)):
            # for batch_idx, batch in enumerate(data_loader):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, e_N)
                his_data = mask_his[e_idxlist[start_idx:end_idx]]
                batch = batch.to(device)
                batch_2 = batch_2.to(device)
                prediction = self.diffusion.p_sample(self.model, batch,batch_2, self.sampling_steps, self.sampling_noise)  # 这里假设你已经定义了 diffusion.p_sample
                # prediction[his_data.nonzero()] = -np.inf

                batch_predictions = prediction.cpu().numpy()

                for user_idx_in_batch in range(len(batch)):
                    global_user_idx = start_idx + user_idx_in_batch

                    excluded_items = excluded_items_per_user[global_user_idx-1]
                    # print(global_user_idx-1)
                    if target_items[global_user_idx]:
                        user_target = target_items[global_user_idx][0]
                        user_predictions = batch_predictions[user_idx_in_batch]

                        all_indices = np.arange(user_predictions.shape[0])
                        exclude_indices = np.setdiff1d(all_indices, excluded_items + [user_target])
                        user_predictions[exclude_indices] = -np.inf

                        # user_prediction_target_first = np.zeros_like(user_predictions)
                        user_prediction_target_first = np.full_like(user_predictions, -np.inf)
                        user_prediction_target_first[0] = user_predictions[user_target]
                        remaining_predictions = user_predictions[np.setdiff1d(all_indices, exclude_indices)]
                        user_prediction_target_first[1:1+len(remaining_predictions)] = remaining_predictions
                        # remaining_predictions = user_predictions[np.setdiff1d(all_indices, exclude_indices+[user_target])]
                        # user_prediction_target_first[1:len(remaining_predictions)] = remaining_predictions
                        predictions.append(user_prediction_target_first)

                    else:
                        # print(f"User {global_user_idx} has no ground truth items. Skipping.")
                        continue

        predictions = np.array(predictions)
        # print(predictions.shape)
        # 假设 finite_counts_per_row 是计算得到的列表
        finite_counts_per_row = np.sum(np.isfinite(predictions), axis=1)
        # print(len(finite_counts_per_row))
        # 检查是否所有元素都相等
        all_equal = np.all(finite_counts_per_row == finite_counts_per_row[0])
        # print(finite_counts_per_row[0])
        # 打印结果
        # if all_equal:
        #     print("All elements in the list are equal.")
        # else:
        #     print("Not all elements in the list are equal.")
        metrics = self.metrics
        
        test_results = self.evaluate_method(predictions, topN, metrics)

        return test_results

    def train(self,data_dict):
        neg_validlist=[]
        neg_testlist=[]
        main_metric_results, dev_results = list(), list()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        device = torch.device("cuda:0" if self.cuda else "cpu")
        for phase in ['train','dev','test']:
            user_ids = data_dict[phase].data['user_id']
            item_ids = data_dict[phase].data['item_id']
            tim = data_dict[phase].data['time']  # 第三列数据，例如评分
            

            data_array = np.column_stack([user_ids, item_ids,tim])
            # 使用 lexsort 进行排序
            sorted_indices = np.lexsort((data_array[:, 2], data_array[:, 0]))  # 先按time 排序，再按 user_id 排序
            sorted_data_array = data_array[sorted_indices]
            result_array = sorted_data_array[:, :2]
            if phase=='train':
                train_list=result_array
            elif phase=='dev':
                valid_list=result_array
                neg1 = data_dict[phase].data['neg_items']
                neg_validlist=neg1
            else:
                test_list=result_array
                neg2 = data_dict[phase].data['neg_items']
                neg_testlist=neg2
            
        uid_max = 0
        iid_max = 0
        train_dict = {}
        
        for uid, iid in train_list:
            if uid not in train_dict:
                train_dict[uid] = []
            train_dict[uid].append(iid)
            if uid > uid_max:
                uid_max = uid
            if iid > iid_max:
                iid_max = iid
        print(len(train_dict.keys()))
        n_user = uid_max +1
        n_item = iid_max +1
        # print(f'user num: {n_user}')
        # print(f'item num: {n_item}')

        
        train_weight = []
        train_list = []
        for uid in train_dict:
            int_num = len(train_dict[uid])
            weight = np.linspace(self.w_min, self.w_max, int_num)
            train_weight.extend(weight)
            for iid in train_dict[uid]:
                train_list.append([uid, iid])
        n_hop = self.n_hop  # The number of hops neighbors, e.g. n_hop=3 means three hops neighbors are taken into account
        print("{}-hop neighbors are taken into account".format(n_hop))
        # if n_hop == 2:
        #     sec_hop = torch.load(self.data_path + 'sec_hop_inters_ML_1M.pt')
        if n_hop == 2:
            sec_hop_file = "{}.pt".format(self.pre_name)  # 使用 str.format() 来动态生成文件名
            sec_hop = torch.load(self.data_path + sec_hop_file)
            multi_hop = sec_hop
        elif n_hop == 3:
            fi="{}.pt".format(self.pre_name)
            multi_hop = torch.load(self.data_path + fi)
        train_list = np.array(train_list)
        # print(len(train_weight))
        # print(len(train_list[:, 0]))
        # print(len(train_list[:, 1]))
        # print(f"最大用户ID: {train_list[:, 0].max()}, 用户总数: {n_user}")
        # print(f"最大物品ID: {train_list[:, 1].max()}, 物品总数: {n_item}")
        train_data = sp.csr_matrix((train_weight, \
                    (train_list[:, 0], train_list[:, 1])), dtype='float64', \
                    shape=(n_user, n_item))

        train_data_ori = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                        (train_list[:, 0], train_list[:, 1])), dtype='float64',
                        shape=(n_user, n_item))

        valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                        (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                        shape=(n_user, n_item))  # valid_groundtruth

        test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                        (test_list[:, 0], test_list[:, 1])), dtype='float64',
                        shape=(n_user, n_item))  # test_groundtruth
        train_dataset = self.DataDiffusion(torch.FloatTensor(train_data.A))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        train_loader_sec_hop = DataLoader(multi_hop, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=0
                                  )
        test_loader_sec_hop = DataLoader(multi_hop, batch_size=self.batch_size, shuffle=False)
        # print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            ### Build Gaussian Diffusion ###
        if self.mean_type == 'x0':
            mean_type = gd.ModelMeanType.START_X
        elif self.mean_type == 'eps':
            mean_type = gd.ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % self.mean_type)

        self.diffusion = gd.GaussianDiffusion(mean_type, self.noise_schedule, \
                self.noise_scale, self.noise_min, self.noise_max, self.steps, device).to(device)

        ### Build MLP ###
        # out_dims = eval(self.dims) + [n_item]
        # in_dims = out_dims[::-1]
        # Build model
        if n_hop == 2:
            self.model = CAM_AE(16, 2, 2, n_item, self.emb_size).to(device)
        elif n_hop == 3:
            self.model = CAM_AE_multihops(16, 4, 2, n_item, self.emb_size).to(device)
        optimizer=self._build_optimizer(self.model)
        # optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print("models ready.")

        self.test_loader=test_loader
        self.test_loader_sec_hop=test_loader_sec_hop
        self.valid_y_data=valid_y_data
        self.train_data=train_data
        self.neg_validlist=neg_validlist
        self.test_y_data=test_y_data
        param_num = 0
        mlp_num = sum([param.nelement() for param in self.model.parameters()])
        diff_num = sum([param.nelement() for param in self.diffusion.parameters()])  # 0
        param_num = mlp_num + diff_num
        print("Number of all parameters:", param_num)
        best_recall, best_epoch = -100, 0
        best_test_result = None
        print("Start training...")
        self._check_time(start=True)
        try:
            for epoch in range(0, self.epoch ):
                # if epoch - best_epoch >= 20:
                #     print('-'*18)
                #     print('Exiting from training early')
                #     break
                self._check_time()
                self.model.train()
                # start_time = time.time()

                batch_count = 0
                total_loss = 0.0
                
                # for batch_idx, batch in enumerate(train_loader):
                #     batch = batch.to(device)
                #     batch_count += 1
                #     optimizer.zero_grad()
                #     losses = self.diffusion.training_losses(self.model, batch, self.reweight)
                #     loss = losses["loss"].mean()
                #     total_loss += loss
                #     loss.backward()
                #     optimizer.step()
                for (batch_idx, batch), (batch_idx_2, batch_2) in zip(enumerate(train_loader), enumerate(train_loader_sec_hop)):
                    batch = batch.to(device)
                    batch_2 = batch_2.to(device)
                    batch_count += 1
                    optimizer.zero_grad()
                    losses = self.diffusion.training_losses(self.model, batch, batch_2, self.reweight)
                    loss = losses["loss"].mean()
                    total_loss += loss
                    loss.backward()
                    optimizer.step()

                training_time = self._check_time()
                dev_result = self.evaluate_imp(test_loader,test_loader_sec_hop, valid_y_data, train_data, self.topN,neg_validlist)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev=({})'.format(
                epoch + 1, loss, training_time, utils.format_metric(dev_result))
                # print(f'dev {logging_str}')
                                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
                    test_result = self.evaluate_imp(test_loader,test_loader_sec_hop, test_y_data, train_data, self.topN,neg_testlist)
                    logging_str += ' test=({})'.format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)
                # print(f'test {logging_str}')
                #Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or \
                        (hasattr(self.model, 'stage') and self.model.stage == 1):
                    
                    utils.check_dir(self.model_path)
                    torch.save(self.model,self.model_path)
                    # self.model.save_model()
                    logging_str += ' *'
                logging.info(logging_str)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
            best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
        # self.model.load_model()
        self.model=torch.load(self.model_path)
        logging.info('Load model from ' + self.model_path)













            
            # if epoch % 5 == 0:
            #     valid_results = self.evaluate(test_loader, valid_y_data, train_data, eval(self.topN))
            #     # if self.tst_w_val:
            #     #     test_results = self.evaluate(test_twv_loader, test_y_data, mask_tv, eval(self.topN))
            #     # else:
            #     test_results = self.evaluate(test_loader, test_y_data, mask_tv, eval(self.topN))
            #     evaluate_utils.print_results(None, valid_results, test_results)
            #     valid_list=list(valid_results.values())
            #     # if valid_results[1][1] > best_recall: # recall@20 as selection
            #     #     best_recall, best_epoch = valid_results[1][1], epoch
            #     #     best_results = valid_results
            #     #     best_test_results = test_results
            #     if valid_list[0] > best_recall: # recall@20 as selection
            #         best_recall, best_epoch = valid_list[0], epoch
            #         best_results = valid_results
            #         best_test_results = test_results
            #         if not os.path.exists(self.save_path):
            #             os.makedirs(self.save_path)
            #         torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
            #             .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
            #             args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))
            
            # print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
            #                     "%H: %M: %S", time.gmtime(time.time()-start_time)))
            # print('---'*18)

        # print('==='*18)
        # print("End. Best Epoch {:03d} ".format(best_epoch))
        # evaluate_utils.print_results(None, best_results, best_test_results)   
        # print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # def print_res(self, dataset: BaseModel.Dataset) -> str:
    #     """
    #     Construct the final result string before/after training
    #     :return: test result string
    #     """
    #     # neg_list=[]
    #     # neg_testlist=[]
    #     # main_metric_results, dev_results = list(), list()
        
        
        
    #     # user_ids = dataset.data['user_id']
    #     # item_ids = dataset.data['item_id']
    #     # tim = dataset.data['time']  # 第三列数据，例如评分
        

    #     # data_array = np.column_stack([user_ids, item_ids,tim])
    #     # # 使用 lexsort 进行排序
    #     # sorted_indices = np.lexsort((data_array[:, 2], data_array[:, 0]))  # 先按time 排序，再按 user_id 排序
    #     # sorted_data_array = data_array[sorted_indices]
    #     # result_array = sorted_data_array[:, :2]

    #     # test_list=result_array
    #     # neg2 = dataset.data['neg_items']
    #     # neg_testlist=neg2
            
    #     # uid_max = 0
    #     # iid_max = 0
    #     # train_dict = {}

    #     # for uid, iid in train_list:
    #     #     if uid not in train_dict:
    #     #         train_dict[uid] = []
    #     #     train_dict[uid].append(iid)
    #     #     if uid > uid_max:
    #     #         uid_max = uid
    #     #     if iid > iid_max:
    #     #         iid_max = iid

    #     # n_user = uid_max 
    #     # n_item = iid_max 
    #     # print(f'user num: {n_user}')
    #     # print(f'item num: {n_item}')

        
    #     # train_weight = []
    #     # train_list = []
    #     # for uid in train_dict:
    #     #     int_num = len(train_dict[uid])
    #     #     weight = np.linspace(self.w_min, self.w_max, int_num)
    #     #     train_weight.extend(weight)
    #     #     for iid in train_dict[uid]:
    #     #         train_list.append([uid, iid])
    #     # train_list = np.array(train_list)
    #     # train_data = sp.csr_matrix((train_weight, \
    #     #             (train_list[:, 0], train_list[:, 1])), dtype='float64', \
    #     #             shape=(n_user, n_item))

    #     # train_data_ori = sp.csr_matrix((np.ones_like(train_list[:, 0]),
    #     #                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
    #     #                 shape=(n_user, n_item))

    #     # valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
    #     #                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
    #     #                 shape=(n_user, n_item))  # valid_groundtruth

    #     # test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
    #     #                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
    #     #                 shape=(n_user, n_item))  # test_groundtruth
    #     # train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
    #     # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    #     # test_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

    #     result_dict = self.evaluate(test_loader, test_y_data, train_data, self.topN,neg_list)
    #     res_str = '(' + utils.format_metric(result_dict) + ')'
    #     return res_str
    
    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    def print_res(self, dataset: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        if self.printflag==0 :
            neg_validlist=[]
            neg_testlist=[]
            main_metric_results, dev_results = list(), list()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
            device = torch.device("cuda:0" if self.cuda else "cpu")
            for phase in ['train','dev','test']:
                user_ids = self.data_dict[phase].data['user_id']
                item_ids = self.data_dict[phase].data['item_id']
                tim = self.data_dict[phase].data['time']  # 第三列数据，例如评分
                

                data_array = np.column_stack([user_ids, item_ids,tim])
                # 使用 lexsort 进行排序
                sorted_indices = np.lexsort((data_array[:, 2], data_array[:, 0]))  # 先按time 排序，再按 user_id 排序
                sorted_data_array = data_array[sorted_indices]
                result_array = sorted_data_array[:, :2]
                if phase=='train':
                    train_list=result_array
                elif phase=='dev':
                    valid_list=result_array
                    neg1 = self.data_dict[phase].data['neg_items']
                    neg_validlist=neg1
                    self.neg_validlist=neg1
                else:
                    test_list=result_array
                    neg2 = self.data_dict[phase].data['neg_items']
                    neg_testlist=neg2
                    self.neg_testlist=neg2
                
            uid_max = 0
            iid_max = 0
            train_dict = {}

            for uid, iid in train_list:
                if uid not in train_dict:
                    train_dict[uid] = []
                train_dict[uid].append(iid)
                if uid > uid_max:
                    uid_max = uid
                if iid > iid_max:
                    iid_max = iid
            # print(len(train_dict.keys()))
            n_user = uid_max +1
            n_item = iid_max +1
            # print(f'user num: {n_user}')
            # print(f'item num: {n_item}')


            train_weight = []
            train_list = []
            for uid in train_dict:
                int_num = len(train_dict[uid])
                weight = np.linspace(self.w_min, self.w_max, int_num)
                train_weight.extend(weight)
                for iid in train_dict[uid]:
                    train_list.append([uid, iid])
            train_list = np.array(train_list)
 
            # print(len(train_weight))
            # print(len(train_list[:, 0]))
            # print(len(train_list[:, 1]))
            # print(f"最大用户ID: {train_list[:, 0].max()}, 用户总数: {n_user}")
            # print(f"最大物品ID: {train_list[:, 1].max()}, 物品总数: {n_item}")
            train_data = sp.csr_matrix((train_weight, \
                        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
                        shape=(int(n_user), int(n_item)))
            self.train_data=train_data
            train_data_ori = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                            (train_list[:, 0], train_list[:, 1])), dtype='float64',
                            shape=(int(n_user), int(n_item)))

            valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                            (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                            shape=(int(n_user), int(n_item)))  # valid_groundtruth
            self.valid_y_data=valid_y_data
            test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                            (test_list[:, 0], test_list[:, 1])), dtype='float64',
                            shape=(int(n_user), int(n_item)))  # test_groundtruth
            self.test_y_data=test_y_data
            train_dataset = self.DataDiffusion(torch.FloatTensor(train_data.A))
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=self.num_workers)
            test_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
            self.test_loader=test_loader
            # print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                ### Build Gaussian Diffusion ###
            if self.mean_type == 'x0':
                mean_type = gd1.ModelMeanType.START_X
            elif self.mean_type == 'eps':
                mean_type = gd1.ModelMeanType.EPSILON
            else:
                raise ValueError("Unimplemented mean type %s" % self.mean_type)

            self.diffusion = gd1.GaussianDiffusion(mean_type, self.noise_schedule, \
                    self.noise_scale, self.noise_min, self.noise_max, self.steps, device).to(device)

            ### Build MLP ###
            out_dims = eval(self.dims) + [int(n_item)]
            in_dims = out_dims[::-1]
            self.model = DNN(in_dims,out_dims, int(self.emb_size), time_type="cat", norm=self.norm).to(device)
            optimizer=self._build_optimizer(self.model)
            # optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            # print("models ready.")

            param_num = 0
            mlp_num = sum([param.nelement() for param in self.model.parameters()])
            diff_num = sum([param.nelement() for param in self.diffusion.parameters()])  # 0
            param_num = mlp_num + diff_num
            # print("Number of all parameters:", param_num)

            result_dict = self.evaluate(test_loader, test_y_data, train_data, self.topN,self.neg_testlist)
        elif self.printflag==1:
            result_dict = self.evaluate_imp(self.test_loader,self.test_loader_sec_hop, self.valid_y_data, self.train_data, self.topN,self.neg_testlist)
        else:
            result_dict = self.evaluate_imp(self.test_loader,self.test_loader_sec_hop, self.test_y_data, self.train_data, self.topN,self.neg_testlist)
        res_str = '(' + utils.format_metric(result_dict) + ')'
        self.printflag=self.printflag+1
        return res_str