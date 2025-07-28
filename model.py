import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm
from layers.DynamicConv import DyConv

seed = 99
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.linear_res = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(out_dim)
        self.linear_3 = nn.Linear(out_dim, 1)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        h = self.dropout(self.linear_2(h))
        res = self.linear_res(x)
        out = self.layernorm(h + res)
        out = self.dropout(out)
        out = self.linear_3(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Conv1d(in_channels=14, out_channels=14, kernel_size=3, stride=1, padding=1)
        self.factor = 1
        self.dropout = 0.05
        self.output_attention = 'store_true'
        self.d_model = 14
        self.n_heads = 8
        self.d_ff = 50
        self.activation = 'gelu'
        self.e_layers = 1
        self.encoder = Encoder(
            [
                EncoderLayer(
                        AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                        output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.DyConv = DyConv(30, 30)
        self.channels = nn.Parameter(torch.rand(14, 14))
        self.fcblock = MLP(420, 256, 64)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        enc_out, attns = self.encoder(x, attn_mask=None)  # [128,30,14]
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.DyConv(enc_out, self.channels)
        enc_out = enc_out.reshape(enc_out.size(0), -1)
        out = self.fcblock(enc_out)
        return out

# 去掉自相关机制
#     def forward(self, x):
#         x = x.permute(0,2,1)
#         x = self.embedding(x)
#         x = x.permute(0,2,1)
#         x = x.permute(0, 2, 1)
#         x = self.DyConv(x, self.channels)
#         x = x.reshape(x.size(0), -1)
#         out = self.fcblock(x)
#         return out

# 去掉动态卷积
#     def forward(self, x):
#         x = x.permute(0,2,1)
#         x = self.embedding(x)
#         x = x.permute(0,2,1)
#         enc_out, attns = self.encoder(x, attn_mask=None) # [128,30,14]
#         enc_out = enc_out.permute(0, 2, 1)
#         enc_out = enc_out.reshape(enc_out.size(0), -1)
#         out = self.fcblock(enc_out)
#         return out
