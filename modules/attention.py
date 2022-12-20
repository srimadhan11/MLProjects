import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    '''
    Global Attention mechanisms:
        * Bahdanau - https://arxiv.org/pdf/1409.0473
        * Luong - https://arxiv.org/pdf/1508.04025
    '''
    def __init__(self, query_size, key_size, g_size=8, alignment=None):
        super().__init__()

        if alignment is None:
            alignment = 'general'

        if alignment == 'concat':
            self.g  = nn.Parameter(torch.empty((g_size, 1)))            # g x 1
            self.Wq = nn.Parameter(torch.empty((g_size, query_size)))   # g x q
            self.Wk = nn.Parameter(torch.empty((g_size,   key_size)))   # g x k
        elif alignment == 'dot':
            pass
        elif alignment == 'general':
            self.W = nn.Parameter(torch.empty((query_size, key_size)))  # q x k
        else:
            raise NotImplementedError(
                f'Unknown alignment model "{alignment}".'
                'Choose from [concat, dot, general]. "general" is default.'
            )

        self.alignment = alignment
        self.align_fun = {
            'concat' : self.__align_concat,
            'dot'    : self.__align_dot,
            'general': self.__align_general,
        }[self.alignment]
        pass

    def __align_concat(self, query, key):
        '''
        a_i = g.T * tanh( W * [q; k_i] )

        Equivalence:
            W * [a; b] == (W1 * a) + (W2 * b)

            where,
                W = [W1, W2]

        NOTE:
            X.T    is transpose of X
            [X; Y] is row    concatenation
            [X, Y] is column concatenation
        '''
        Bq,     q = query.shape
        Bk, Lk, k = key  .shape
        assert (Bq == Bk, 'batch size mismatch')

        Q = torch.matmul(self.Wq, query.unsqueeze(2))           # B x g x 1
        K = torch.matmul(self.Wk,   key.unsqueeze(3))           # B x L x g x 1
        additive = Q.unsqueeze(1) + K                           # B x L x g x 1
        product  = torch.matmul(self.g.T, torch.tanh(additive)) # B x L x 1 x 1
        return product.squeeze()                                # B x L

    def __align_dot(self, query, key):
        '''
        a_i = q * k_i
        '''
        Bq,     q = query.shape
        Bk, Lk, k = key  .shape
        assert (Bq == Bk, 'batch size mismatch')
        assert ( q ==  k,   'dim size mismatch')

        product = torch.matmul(
            query.reshape(Bq, 1 , 1, q).expand(-1, Lk, -1, -1),
            key  .reshape(Bk, Lk, k, 1)
        )                               # B x L x 1 x 1
        return product.squeeze()        # B x L

    def __align_general(self, query, key):
        '''
        a_i = q * W * k_i
        '''
        Bq,     q = query.shape
        Bk, Lk, k = key  .shape
        assert (Bq == Bk, 'batch size mismatch')

        product = torch.matmul(
            query.reshape(Bq, 1 , 1, q).expand(-1, Lk, -1, -1),
            self.W
        )                               # B x L x 1 x k
        product = torch.matmul(
            product,
            key.reshape(Bk, Lk, k, 1)
        )                               # B x L x 1 x 1
        return product.squeeze()        # B x L

    def forward(self, query, key, value, return_weights=False):
        '''
        Parameter
            query: B x q
            key  : B x L x k
            value: B x L x k
        '''
        assert(query.shape[0] == value[0].shape, 'batch size mismatch')
        assert(key.shape == value.shape, 'key and value size mismatch')

        scores  = self.align_fun(query, key)                            # B x L
        weights = F.softmax(scores, dim=1)                              # B x L
        context = torch.bmm(weights.unsqueeze(1), value).squeeze(1)     # B x k

        if return_weights:
            return context, weights
        return context
    pass
