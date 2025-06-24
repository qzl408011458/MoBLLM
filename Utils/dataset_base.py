from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader
from Utils.MatrixADJ_Build import Info_Station

class TripDataset(Dataset):
    ########################
    # ts week
    # t time point in a day
    # o begin station
    # d destination station
    ########################
    def __init__(self, data_list, ):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        input = self.data_list[i][0]
        label = self.data_list[i][1]

        return input, label




