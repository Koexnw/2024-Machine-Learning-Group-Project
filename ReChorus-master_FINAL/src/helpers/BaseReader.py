# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils
from openpyxl import load_workbook

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
    # def _read_data(self):
    #     logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
    #     self.data_df = dict()
    #     for key in ['train', 'dev', 'test']:
    #         csv_path = os.path.join(self.prefix, self.dataset, key + '.csv')
    #         excel_path = os.path.join(self.prefix, self.dataset, key + '.xlsx')  # Try .xlsx first
    #         if not os.path.exists(csv_path):
    #             if os.path.exists(excel_path):
    #                 try:
    #                     #Use openpyxl to read the excel file.  This handles both .xlsx and .xls, generally.
    #                     wb = load_workbook(excel_path, data_only=True) # data_only=True to get values, not formulas
    #                     sheet = wb.active #Get the active sheet
    #                     data = list(sheet.values) # Convert to list of lists/rows
    #                     headers = data[0]
    #                     df_data = data[1:]
    #                     self.data_df[key] = pd.DataFrame(df_data, columns=headers).reset_index(drop=True).sort_values(by=['user_id', 'time'])
    #                     logging.warning(f"CSV not found for {key}, using Excel file instead.")

    #                 except Exception as e:
    #                     logging.error(f"Error reading Excel file {excel_path}: {e}")
    #                     raise  # Re-raise the exception to stop execution
    #             else:
    #                 logging.error(f"Could not find CSV or Excel file for {key} dataset.")
    #                 raise FileNotFoundError(f"Neither CSV nor Excel file found for {key} at {csv_path} or {excel_path}")

    #         else:
    #             self.data_df[key] = pd.read_csv(csv_path, sep=self.sep).reset_index(drop=True).sort_values(by=['user_id', 'time'])

    #         self.data_df[key] = utils.eval_list_columns(self.data_df[key])

    #     logging.info('Counting dataset statistics...')
    #     key_columns = ['user_id', 'item_id', 'time']
    #     if 'label' in self.data_df['train'].columns:  # Add label for CTR prediction
    #         key_columns.append('label')
    #     self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
    #     self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
    #     for key in ['dev', 'test']:
    #         if 'neg_items' in self.data_df[key]:
    #             neg_items = np.array(self.data_df[key]['neg_items'].tolist())
    #             assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
    #     logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
    #         self.n_users - 1, self.n_items - 1, len(self.all_df)))
    #     if 'label' in key_columns:
    #         positive_num = (self.all_df.label == 1).sum()
    #         logging.info('"# positive interaction": {} ({:.1f}%)'.format(
    #             positive_num, positive_num / self.all_df.shape[0] * 100))
        
