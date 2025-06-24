import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeepMove(nn.Module):
    def __init__(self, rnnUnit, st_sum, hid_dim_rnn, embed_size=20, task='ori',
                 device='cuda'):
        super(DeepMove, self).__init__()
        self.task = task
        self.device = device
        self.embed_size = embed_size
        self.rnnUnit = rnnUnit
        self.hid_dim_rnn = hid_dim_rnn
        self.st_sum = st_sum
        self.__init_rnnModules(rnnUnit, hid_dim_rnn)
        self.__init_EmbedModules(st_sum)
        self.__init_pred_module(hid_dim_rnn, st_sum)

    def __init_EmbedModules(self, st_sum):
        self.emb_sloc = nn.Embedding(st_sum + 2, self.embed_size)
        self.emb_eloc = nn.Embedding(st_sum + 2, self.embed_size)
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


    
    def __init_rnnModules(self, rnnUnit, hid_dim_rnn):
        if rnnUnit == 'RNN':
            self.encoder = nn.RNN(self.embed_size * 2 + 20, hid_dim_rnn, batch_first=True) # 17
            self.decoder = nn.RNN(self.embed_size * 2 + 20, hid_dim_rnn, batch_first=True)
        if rnnUnit == 'LSTM':
            self.encoder = nn.LSTM(self.embed_size * 2 + 20, hid_dim_rnn, batch_first=True)
            self.decoder = nn.LSTM(self.embed_size * 2 + 20, hid_dim_rnn, batch_first=True)
        if rnnUnit == 'GRU':
            self.encoder = nn.GRU(self.embed_size * 2 + 20, hid_dim_rnn, batch_first=True)
            self.decoder = nn.GRU(self.embed_size * 2 + 20, hid_dim_rnn, batch_first=True)
        self.attention = Attention(hid_dim_rnn)

    def __init_pred_module(self, hid_dim_rnn, st_sum):
        self.pred_o = nn.Sequential(
                nn.Linear(hid_dim_rnn * 2, hid_dim_rnn * 2),
                nn.ReLU(),
                nn.Linear(hid_dim_rnn * 2, st_sum + 1)
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


        input_enc = torch.cat([out_emb_sloc, out_emb_eloc, out_emb_smin,
                               out_emb_dura, out_emb_w], dim=-1)

        input_dec = torch.cat([out_emb_sloc_tar, out_emb_eloc_tar, out_emb_smin_tar,
                               out_emb_dura_tar, out_emb_w_tar], dim=-1)

        if self.rnnUnit == 'LSTM':
            output_enc, (h_enc, c_enc) = self.encoder(input_enc)
            output_dec, (h_dec, c_dec) = self.decoder(input_dec, (h_enc, c_enc))
        if self.rnnUnit == 'GRU' or self.rnnUnit == 'RNN':
            output_enc, h_enc = self.encoder(input_enc)
            output_dec, h_dec = self.decoder(input_dec, h_enc)

        encoder_outputs = output_enc.transpose(0, 1)#transpose:change x and y axis

        attn_weights = self.attention(h_dec, encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        out_f = torch.cat([h_dec, context], dim=-1).squeeze(0)

        out_eloc = self.pred_o(out_f)

        return out_eloc

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]




