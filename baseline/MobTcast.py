import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(0)].clone().detach().transpose(0, 1)



class MobTcast(nn.Module):
    def __init__(self, st_sum, model_dim, nhead=8, layers=1, task='ori',
                 device='cuda'):
        super(MobTcast, self).__init__()
        self.task = task
        self.embed_size = 32
        self.device = device
        self.st_sum = st_sum
        emb_all_dims = 32 + 32 + 8 + 8 + 4

        self.__init_EmbedModules()
        self.lin2enc = nn.Linear(emb_all_dims, model_dim, bias=False)

        self.__init_encoder(model_dim, nhead, layers)

        self.pos_encoder = PositionalEncoding(model_dim, n_position=100)

    def __init_EmbedModules(self):
        self.emb_sloc = nn.Embedding(self.st_sum + 2, self.embed_size)
        self.emb_eloc = nn.Embedding(self.st_sum + 2, self.embed_size)
        self.ph_eloc = nn.Parameter(torch.randn(self.embed_size))

        self.emb_w = nn.Embedding(7 + 1, 4)

        self.emb_smin = nn.Sequential(
            nn.Linear(1, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 8, bias=False),
        )
        # self.ph_smin = torch.randn()

        self.emb_dura = nn.Sequential(
            nn.Linear(1, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 8, bias=False),
        )
        if self.task == 'ori':
            self.ph_dura = nn.Parameter(torch.randn(8))

    
    def __init_encoder(self, mdim, nhead, layers):
        encoder_layer = nn.TransformerEncoderLayer(d_model=mdim, nhead=nhead, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.rnn_hid2out_d = nn.Sequential(
                nn.Linear(mdim, mdim * 2),
                nn.ReLU(),
                nn.Linear(mdim * 2, self.st_sum + 1)
        )


    def forward(self, x, y_tar):
        out_emb_sloc = self.emb_sloc(x[0])
        out_emb_eloc = self.emb_eloc(x[1])
        out_emb_smin = self.emb_smin(x[2])
        out_emb_dura = self.emb_dura(x[3])
        out_emb_w = self.emb_w(x[4])

        out_emb_sloc_tar = self.emb_sloc(y_tar[0])
        out_emb_eloc_tar = self.ph_eloc.repeat((out_emb_eloc.shape[0], 1, 1))
        out_emb_smin_tar = self.emb_smin(y_tar[1])
        if self.task == 'ori':
            out_emb_dura_tar = self.ph_dura.repeat((out_emb_dura.shape[0], 1, 1))
            out_emb_w_tar = self.emb_w(y_tar[2])
        if self.task == 'dest':
            out_emb_dura_tar = self.emb_dura(y_tar[2])
            out_emb_w_tar = self.emb_w(y_tar[3])

        input_enc1 = torch.cat([out_emb_sloc, out_emb_eloc, out_emb_smin,
                               out_emb_dura, out_emb_w], dim=-1)

        input_enc2 = torch.cat([out_emb_sloc_tar, out_emb_eloc_tar, out_emb_smin_tar,
                               out_emb_dura_tar, out_emb_w_tar], dim=-1)

        input_enc = torch.cat([input_enc1, input_enc2], dim=1)
        input_enc = self.lin2enc(input_enc).transpose(0, 1)

        enc_output = self.encoder(self.pos_encoder(input_enc))

        output_d = self.rnn_hid2out_d(enc_output[-1])


        return output_d





