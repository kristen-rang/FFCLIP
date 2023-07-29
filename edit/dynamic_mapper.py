from enum import Enum
import torch
import torch.nn as nn
from stylegan2.model import PixelNorm, Generator
from torch.nn import Module, LayerNorm, LeakyReLU
from edit.modules import attention_mapping

class att_module(Module):
    def __init__(self, latent_dim=512):
        super(att_module, self).__init__()
        self.pixelnorm = PixelNorm()
        #self.norm2d = LayerNorm([18, latent_dim], elementwise_affine=False)
        self.norm1d = LayerNorm([latent_dim], elementwise_affine=False)
        self.q_matrix = nn.Linear(512, 512, bias=False)
        self.k_matrix = nn.Linear(512, 512, bias=False)
        self.v_matrix = nn.Linear(512, 512, bias=False)

        self.gamma_ = nn.Sequential(nn.Linear(512, 512), LayerNorm([512]), LeakyReLU(), nn.Linear(512, 512))
        self.beta_ = nn.Sequential(nn.Linear(512, 512), LayerNorm([512]), LeakyReLU(), nn.Linear(512, 512))
        self.fc = nn.Linear(512, 512)
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = LeakyReLU()
        self.attention_layer = attention_mapping(in_dim=18)


    def forward(self, x, embd):
        '''
        :param x: B,18,512
        :param embd: B,1,512
        :return:
        '''
        x = self.pixelnorm(x)
        K = self.k_matrix(x)  # B,18,512
        V = self.v_matrix(x)  # B,18,512
        Q = self.q_matrix(embd).permute(0, 2, 1)  # B,1,512 -> B,512,1

        score = torch.matmul(K, Q)
        score = self.softmax(score)  # B, 18, 1

        h = score * V  # importance choice
        t = self.attention_layer(x, embd)  # translation in layer dim

        h = h + t

        h = self.norm1d(h)
        gamma = self.gamma_(embd)
        beta = self.beta_(embd)


        h = h * (1 + gamma) + beta
        h = self.leakyrelu(h)

        return h


class Dynamic_mapper(Module):
    def __init__(self):
        super(Dynamic_mapper, self).__init__()
        self.att_mapper = nn.ModuleList([att_module() for _ in range(4)])

    def forward(self, x, embd):
        for mapper in self.att_mapper:
            x = mapper(x, embd)
        return x


class EditModel(Module):
    def __init__(self, opts):
        super(EditModel, self).__init__()
        self.configs = opts
        self.generator = Generator(style_dim=512,n_mlp=8,size=1024,channel_multiplier=2,blur_kernel=[ 1, 3, 3, 1 ],lr_mlp=0.01)
        self.load_pretrain(self.configs)
        self.att_mapper = Dynamic_mapper()

    def load_pretrain(self, configs):
        ckpt = torch.load(configs.pretrain_path, map_location=torch.device('cpu'))  ##########
        self.generator.load_state_dict(ckpt['g_ema'], strict=True)
        self.generator.latent_avg = ckpt['latent_avg']
        # eval()
        self.generator = self.generator.eval()

        # freeze parameters
        for param in self.generator.parameters():
            param.requires_grad = False

    def forward(self, w, attributes):
        '''
        :param x: img from batch B,1024,1024,3
        :param attributes: B,512
        :return:
        '''
        deta_w = self.att_mapper(w, attributes)
        w_hat = w + deta_w

        return w_hat  # w: B,18,512 w_hat: B




