
import torch
from torch import nn
import pdb
from torch.nn import functional as F

class AttentionModel(nn.Module):
    """ linear - tanh - linear - softmax - einsum """

    def __init__(self, D_in, D_out = 1, bias = True, normalize_F = True, is_att=True):
        """
        Args:
            D_in (int): Input dimension
            D_out (int): Output dimension
        """
        super(AttentionModel, self).__init__()
        self.is_att = is_att
        self.normalize_F = normalize_F
        self.linear1 = nn.Linear(in_features = D_in,
                                 out_features = D_in,
                                 bias=bias)
        self.linear2 = nn.Linear(in_features = D_in,
                                 out_features = D_out,
                                 bias=bias)
        
        print('.'*30)
        print('config attention')
        if self.normalize_F:
            print('normalize_F')
        else:
            print('no constraint F')
        print('.'*30)

    def forward(self, x):
        """
        Args:
            x = vgg feature input
        """
        if self.is_att:
            ###
            if self.normalize_F:
                x = F.normalize(x,dim = -1)  
            ###
            
            h_tanh = F.tanh(self.linear1(x))
            output = self.linear2(h_tanh)
            output = output[:,:,:,0]
        else:
            output = x.new_ones((x.size()[:-1]))
            
        alphas = F.softmax(output, dim = 2)
        fbar = torch.einsum('bnr, bnrd -> bnd', alphas, x)    #b:batch_size, n:#frames, r:49, d:512
        return fbar,alphas #bnd,bnr
