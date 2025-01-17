import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class CH_Diffrec(GeneralModel):
    reader = 'BaseReader'
    runner = 'Diffirunner'


    class Dataset(GeneralModel.Dataset):
        # No need to sample negative items
        # def actions_before_epoch(self):
        #     self.data['neg_items'] = [[] for _ in range(len(self))]
        def ik(self):
            a=1
