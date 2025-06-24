from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader
from Utils.MatrixADJ_Build import Info_Station

class TripsDataset(Dataset):
    ########################
    # ts week
    # t time point in a day
    # o begin station
    # d destination station
    ########################
    def __init__(self, ts_tensor, t_tensor, o_tensor, d_tensor,
                 o_ent_tensor, d_ent_tensor, o_prob_tensor, d_prob_tensor, matrix_adjacent, device='cuda'):
        # Each tensor shape (N, timestep)
        # Forward 69 columns are input, last column is label
        self.device = device
        self.ts_inputs = ts_tensor
        self.t_inputs = t_tensor
        #self.o_inputs, self.o_labels = o_tensor.split(o_tensor.size(-1)-1, dim=-1)
        self.o_inputs = o_tensor
        self.d_inputs, self.d_labels = d_tensor.split(d_tensor.size(-1)-1, dim=-1)
        self.o_ent_inputs = o_ent_tensor
        self.d_ent_inputs = d_ent_tensor
        self.o_prob_inputs = o_prob_tensor.unsqueeze(-1)
        self.d_prob_inputs = d_prob_tensor.unsqueeze(-1)

        self.labels = self.d_labels.squeeze()
        self.t_inputs = (self.t_inputs * 24).to(torch.int64)
        self.device = device
        self.nodes_num = len(matrix_adjacent)
        # self.graph = torch.tensor([matrix_adjacent], dtype=torch.float, device=device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item].to(self.device)
        #print(self.o_inputs.shape)
        input = {
            'w': self.ts_inputs[item, :].to(self.device),
            't': self.t_inputs[item, :].to(self.device),
            'o': self.o_inputs[item, :].to(self.device),
            'd': self.d_inputs[item, :].to(self.device),
            # 'o_ent': self.o_ent_inputs[item, :].to(self.device),
            # 'd_ent': self.d_ent_inputs[item, :].to(self.device),
            # 'o_prob': self.o_prob_inputs[item, :].to(self.device),
            # 'd_prob': self.d_prob_inputs[item, :].to(self.device)
        }
        return input, label

'''
    def T_od2tripVector(self, tensor_o, tensor_d):
        # input tensor shape (samples, timestep)
        # output tensor shape (samples, timestep, nodes_num, 2), 2 means (in, out)
        # Transfer tensors o and d to trip vector as input of graph model
        oh_tensor_o = F.one_hot(tensor_o, num_classes=self.nodes_num).unsqueeze(-1).float()
        oh_tensor_d = F.one_hot(tensor_d, num_classes=self.nodes_num).unsqueeze(-1).float()
        # Leave o_st for d_st
        return torch.cat([oh_tensor_o, oh_tensor_d], dim=-1)
'''

if __name__ == '__main__':
    with open('../datasetAG/dataset_AG_d3_len40_o.pkl', 'rb') as fr:
        w_train, t_train, o_train, d_train, o_train_fre, d_train_fre, o_train_ent, d_train_ent, \
        w_test, t_test, o_test, d_test, o_test_fre, d_test_fre, o_test_ent, d_test_ent = pickle.load(fr)

    dataset_train = TripsDataset(w_train, t_train, o_train, d_train, o_train_ent, d_train_ent,
                                 o_train_fre, d_train_fre, Info_Station().matrix_adjacent, device='cpu')
    trainloader = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True)
    for step, (batch_x, batch_y) in enumerate(trainloader):
        print(batch_x['o'].shape, batch_x['d'].shape)
        print(batch_x['o_prob'].shape, batch_x['d_prob'].shape)
        break



