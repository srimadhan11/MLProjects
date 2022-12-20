import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    '''
    Label Smoothing with KL-divergence loss
    '''
    def __init__(self, tgt_vocab_size, ignore_index, label_smoothing=0.0) -> None:
        super().__init__()
        assert 0.0 < label_smoothing <= 1.0

        smoothing_value          = label_smoothing / (tgt_vocab_size - 2)
        one_hot                  = torch.full((1, tgt_vocab_size), smoothing_value)
        one_hot[:, ignore_index] = 0
        self.register_buffer(name='one_hot', tensor=one_hot)

        self.confidence   = 1.0 - label_smoothing
        self.ignore_index = ignore_index
        pass

    def forward(self, output, target):
        '''
        Parameters:
            output: FloatTensor
            target: LongTensor
        '''
        batch_output, vocab_size = output.shape
        batch_target,            = target.shape
        assert batch_output == batch_target

        target = target.unsqueeze(1)

        true_dist = self.one_hot.expand(batch_target, -1).clone()
        true_dist.scatter_(1, target, self.confidence)
        true_dist.masked_fill_((target == self.ignore_index), 0)

        return F.kl_div(output, true_dist, reduction='sum')
    pass
