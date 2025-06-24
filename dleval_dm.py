import copy

import torch
import torch.nn as nn
import pickle
import time

from sklearn.metrics import f1_score, mean_absolute_error
from tqdm import tqdm

from Utils.MatrixADJ_Build import Info_Station
from baseline.DeepMove import DeepMove
from Utils.dataset_base import TripDataset
from torch.utils.data import DataLoader
import argparse
import os

import numpy as np
import random

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def getL2regularization(params, device='cuda'):
    l2_regularization = torch.tensor(0.0, device=device)
    for param in params:
        l2_regularization += torch.norm(param, 2)
    return l2_regularization


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.loss_clc = nn.CrossEntropyLoss()

    def forward(self, pred, true):
        #print(pred.shape, true.shape)
        loss0 = self.loss_clc(pred, true)

        return loss0


def evaluation(pred, true):
    pred_list, true_list = [], []


    for i in range(len(pred)):
        pred_list.append(pred[i])
        true_list.append(true[i])

    # torch.argmax(dim=-1)
    pred, true = np.array(torch.cat(pred_list).argmax(dim=-1).to('cpu')), np.array(torch.cat(true_list).to('cpu'))

    eval_acc = np.sum(pred == true) / len(pred)
    eval_f1 = f1_score(true, pred, average='weighted')

    return eval_acc, eval_f1

def model_save(saveName):
    rootpath = args.save_path + '/'
    global date, best_model_path
    if os.path.exists(rootpath + date + '_' + model_name) is False:
        os.makedirs(rootpath + date + '_' + model_name)
    best_model_path = rootpath + date + '_' + model_name + '/' + saveName + '.pkl'
    torch.save(model.state_dict(), best_model_path)

def record_tofile(str):
    global date
    dirpath = args.save_path + '/' + date + '_' + model_name + '/'
    global record_file
    if record_file is None:
        record_file = open(dirpath + 'train_record.txt', 'w', encoding='utf-8')
    record_file.write(str + '\n')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def test():
    global model_name, best_model_path
    #model.eval()
    print(best_model_path)
    model.load_state_dict(torch.load(best_model_path))
    start_time = time.time()
    test_pred_y_list = []
    test_true_y_list = []
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(testloader):
            if args.task == 'ori':
                y_tar = [batch_y[0], batch_y[2].unsqueeze(dim=-1), batch_y[4].unsqueeze(dim=-1)]
            if args.task == 'dest':
                y_tar = [batch_y[0], batch_y[2].unsqueeze(dim=-1), batch_y[3].unsqueeze(dim=-1),
                         batch_y[4].unsqueeze(dim=-1)]
            batch_y = batch_y[1][:, 0]
            test_pred_y = model(batch_x, y_tar)
            test_pred_y_list.append(test_pred_y)
            test_true_y_list.append(batch_y)

            print('\rTesting, batch: {:d}/{:d}'.format(step + 1, len(testloader)), end='')
    print()
    accuracy, f1 = evaluation(test_pred_y_list, test_true_y_list)

    end_time = time.time()

    print('**********************************************')
    print('Test finished. Time: {:2f}\n'
          'acc_o: {:2f}\n'.format(end_time - start_time, accuracy * 100
                                  ),
          'f1_o: {:4f}\n'.format(f1)
          )


    try:
        if is_saveModel:
            record = 'Test, acc_o: {:2f}, f1_o: {:4f}'.format(
                accuracy, f1
            )
            record_tofile(record)
    except:
        print('Writing results to record was failed!!!!!!')

def test2():
    global model_name, best_model_path
    #model.eval()
    print(best_model_path)
    model.load_state_dict(torch.load(best_model_path))
    start_time = time.time()
    test_pred_y_list = []
    test_true_y_list = []
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(testloader):
            if args.task == 'ori':
                y_tar = [batch_y[0], batch_y[2].unsqueeze(dim=-1), batch_y[4].unsqueeze(dim=-1)]
            if args.task == 'dest':
                y_tar = [batch_y[0], batch_y[2].unsqueeze(dim=-1), batch_y[3].unsqueeze(dim=-1),
                         batch_y[4].unsqueeze(dim=-1)]
            batch_y = batch_y[1][:, 0]
            test_pred_y = model(batch_x, y_tar)
            test_pred_y_list.append(test_pred_y)
            test_true_y_list.append(batch_y)

            print('\rTesting, batch: {:d}/{:d}'.format(step + 1, len(testloader)), end='')
    print()
    accuracy, f1 = evaluation(test_pred_y_list, test_true_y_list)

    end_time = time.time()

    print('**********************************************')
    print('Test finished. Time: {:2f}\n'
          'acc_o: {:2f}\n'.format(end_time - start_time, accuracy * 100
                                  ),
          'f1_o: {:4f}\n'.format(f1)
          )



def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def co_fn(batch):
    max_len = 0
    feat_len = 0
    for _ in batch:
        if len(_[0]) > max_len:
            max_len = len(_[0])
        if len(_[1][0]) > feat_len:
            feat_len = len(_[1][0])

    new_inputs = []
    labels = []
    # padding = 0
    pad_list = [0, 0, -1, -1, 0]

    # on feature dimension: o, d, t_o, t_d, week
    for bat in batch:
        input, label = bat
        pad_arr = np.array([pad_list for __ in range(max_len - len(input))])
        if len(pad_arr):
            new_input = np.concatenate([pad_arr, input + 1]).reshape(1, -1, feat_len)
        else:
            new_input = input.reshape(1, -1, feat_len)
        new_inputs.append(new_input)
        labels.append(label)
    new_inputs = np.concatenate(new_inputs)
    input_o = torch.tensor(new_inputs[:, :, 3], device=args.device, dtype=torch.long)#.reshape(len(new_inputs), max_len, 1)
    input_d = torch.tensor(new_inputs[:, :, 4], device=args.device, dtype=torch.long)#.reshape(len(new_inputs), max_len, 1)
    input_to = torch.tensor(new_inputs[:, :, 0:1], device=args.device, dtype=torch.float32)#.reshape(len(new_inputs), max_len, 1)
    input_td = torch.tensor(new_inputs[:, :, 2:3], device=args.device, dtype=torch.float32)#.reshape(len(new_inputs), max_len, 1)
    input_w = torch.tensor(new_inputs[:, :, 1], device=args.device, dtype=torch.long)#.reshape(len(new_inputs), max_len, 1)
    # print()

    new_labels = np.concatenate(labels)
    label_o = torch.tensor(new_labels[:, 3:4], device=args.device, dtype=torch.long)
    label_d = torch.tensor(new_labels[:, 4:5], device=args.device, dtype=torch.long)
    label_to = torch.tensor(new_labels[:, 0:1], device=args.device, dtype=torch.float32)
    label_td = torch.tensor(new_labels[:, 2:3], device=args.device, dtype=torch.float32)
    label_w = torch.tensor(new_labels[:, 1], device=args.device, dtype=torch.long)

    return ((input_o, input_d, input_to, input_td, input_w),
            (label_o, label_d, label_to, label_td, label_w))



def init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='HK Metro Trip Prediction Model Train')
    parser.add_argument('-rnn', default='GRU', type=str,
                        help='Choose LSTM or GRU as model backbone')
    parser.add_argument('-gcn', default='GCN', type=str,
                        help='Select GCN, GAT, Cheb as the spatial features extracting module')
    parser.add_argument('-in_dim', default=2, type=int,
                        help='GCN module input dims, denote leave station_out(0, 1) for station_in(1, 0)')
    parser.add_argument('-stations_sum', default=91, type=int,
                        help='The sum of the stations in the model')
    parser.add_argument('-rnn_hid_dim', default=256, type=int,
                        help='Hidden size of RNN backbone')
    parser.add_argument('-embed_size', default=32, type=int,
                        help='Hidden size of RNN backbone')
    parser.add_argument('-device', default='cuda:1', type=str,
                        help='If GPU is used')
    parser.add_argument('-batch_size', default=32, type=int,
                        help='Batch size of training model ')
    parser.add_argument('-save_path', default='modelSave_dl', type=str,
                        help='A content path for model saving')
    parser.add_argument('-attention', default=True, type=str2bool,
                        help='If use attention module')
    parser.add_argument('-saveModel', default=True, type=str2bool,
                        help='If save model')
    parser.add_argument('-seq', default=41, type=int,
                        help='Choose sequence length of real data')
    parser.add_argument('-task', default='ori', type=str,
                        help='Choose task as ori or dest')
    parser.add_argument('-data', default='hk_network_change', type=str,
                        help='')

    global args
    args = parser.parse_args(argv)
# python train_dm.py -task ori -data hk -device cuda:0
# python train_dm.py -task dest -data hk -device cuda:1
# python train_dm.py -task ori -data hz -device cuda:2
# python train_dm.py -task dest -data hz -device cuda:3

def normalize(train_sample_list, test_list, mask=None):
    trip_data = []
    for i in range(len(train_sample_list)):
        trip_data.append(train_sample_list[i][0])
        trip_data.append(train_sample_list[i][1])
    trip_data = np.concatenate(trip_data)

    mean = trip_data.mean(axis=0, keepdims=True)[0]
    std = trip_data.std(axis=0, keepdims=True)[0]
    scalar = [(mean[_], std[_]) for _ in range(len(mean))]
    for data_list in [train_sample_list, test_list]:
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                for k in range(len(mask)):
                    if mask[k]:
                        data_list[i][j][:, k] = (data_list[i][j][:, k] - mean[k]) / std[k]

    return train_sample_list, test_list, scalar

def denormalize(data, scalar):
    mean, std = scalar
    return data * std + mean

def data_prepare(data_dic):
    data_list = []
    for uid in data_dic:
        if len(data_dic[uid]) != 0:
            for i in range(len(data_dic[uid])):
                new_data_list = [
                    np.concatenate([data_dic[uid][i][0], data_dic[uid][i][1]], axis=0),
                    data_dic[uid][i][2]
                ]
                data_list.append(new_data_list)
        # data_list.extend(data_dic[uid])
    return data_list

if __name__ == '__main__':
    global args
    seed = int(random.randint(1, 100000))
    seed_torch(seed)
    parse_args()
    EPOCHES = 300
    PATIENCE = 10
    patience = 0
    matrix_adjacent = Info_Station().matrix_adjacent
    record_file = None

    best_model_path = 'modelSave_dl/2025-06-06 15-50-56_dm_base_GRU,len41,embed_size32,hid_dim256,seed89127,dest,hk/epoch3.pkl'
    args.task = 'dest' # or "ori"
    args.data = 'hk_incident' # or "hk_policy_intervention", "hk_event", "hk_network_change"

    # only for evaluating the special scenarios
    if args.data == 'hk_network_change':
        with open(f'data4eval/{args.task}_network_change.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)

    if args.data == 'hk_policy_intervention':
        with open(f'data4eval/{args.task}_policy_intervention.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)

    if args.data == 'hk_event':
        with open(f'data4eval/{args.task}_original_event.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)

    if args.data == 'hk_incident':
        with open(f'data4eval/{args.task}_original_incident.pkl', 'rb') as fr:
            train_data, valid_data, test_data = pickle.load(fr)


    mask = [1, 0, 1, 0, 0]
    train_sample_list, valid_list, test_list = data_prepare(train_data), data_prepare(valid_data), data_prepare(test_data)

    train_sample_list0 = copy.deepcopy(train_sample_list)
    train_list, test_list, scalar = normalize(train_sample_list0, test_list, mask=mask)
    _, valid_list, _ = normalize(train_sample_list0, valid_list, mask=mask)


    test_set = TripDataset(test_list)


    model_name = f'dm_base_{args.rnn},len{args.seq},embed_size{args.embed_size},' \
                 f'hid_dim{args.rnn_hid_dim},seed{seed},{args.task},{args.data}'

    is_saveModel = args.saveModel
    device = args.device
    best_acc = 0.0
    date = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())


    testloader = DataLoader(dataset=test_set, batch_size=args.batch_size, collate_fn=co_fn)

    criterion = Criterion()

    model = DeepMove(rnnUnit=args.rnn, st_sum=args.stations_sum,
                         embed_size=args.embed_size, hid_dim_rnn=args.rnn_hid_dim,
                         device=device, task=args.task).to(device)

    test()



