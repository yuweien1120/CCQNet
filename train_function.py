import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.modules import loss
import torch.nn.functional as F

'''
some train functions for different models
'''




# region QFAE

def group_parameters(m):
    # group_r = list(map(lambda x: x[1], list(filter(lambda kv: '_r' in kv[0], m.named_parameters()))))
    # group_g = list(map(lambda x: x[1], list(filter(lambda kv: '_g' in kv[0], m.named_parameters()))))
    # group_b = list(map(lambda x: x[1], list(filter(lambda kv: '_b' in kv[0], m.named_parameters()))))
    # group_bias = list(map(lambda x: x[1], list(filter(lambda kv: 'bias' in kv[0], m.named_parameters()))))
    weight_r, weight_g, weight_b, bias_r, bias_g, bias_b, w, b = [], [], [], [], [], [], [], []
    for name, p in m.named_parameters():
        if 'weight_r' in name:
            weight_r += [p]
        if 'weight_g' in name:
            weight_g += [p]
        if 'weight_b' in name:
            weight_b += [p]
        if 'bias_r' in name:
            bias_r += [p]
        if 'bias_g' in name:
            bias_g += [p]
        if 'bias_b' in name:
            bias_b += [p]
        if 'weight' in name[-6:]:
            w += [p]
        if 'bias' in name[-4:]:
            b += [p]
    return (weight_r, weight_g, weight_b, bias_r, bias_g, bias_b, w, b)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()