import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self, Q, K, V, temperature=None, mask=None):
        '''
        Parameters:
            Q: [b x l_q x d_k]
            K: [b x l_k x d_k]
            V: [b x l_k x d_v]
        '''
        b_q, l_q, d_q = Q.shape
        b_k, l_k, d_k = K.shape
        b_v, l_v, d_v = V.shape

        assert b_q == b_k == b_v
        assert l_k == l_v
        assert d_q == d_k

        if temperature is None:
            temperature = math.sqrt(d_k)

        score = torch.bmm(Q, K.transpose(1, 2)) / temperature

        if mask is not None:
            score = score.masked_fill(mask, -1E9)

        score     = self.dropout(F.softmax(score, dim=2))
        attention = torch.bmm(score, V)
        return attention
    pass


class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, d_model, dropout=0.1, d_k=None, d_v=None, bias=False) -> None:
        super().__init__()
        assert d_model % nhead == 0

        self.nhead   = nhead
        self.d_model = d_model

        if d_k is None:
            d_k = d_model // nhead
        if d_v is None:
            d_v = d_k

        self.d_k = d_k
        self.d_v = d_v

        self.dropout   = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.W_Q = nn.Linear(d_model, nhead * d_k, bias=bias)
        self.W_K = nn.Linear(d_model, nhead * d_k, bias=bias)
        self.W_V = nn.Linear(d_model, nhead * d_v, bias=bias)
        self.W_O = nn.Linear(nhead * d_v, d_model, bias=bias)
        pass

    def forward(self, Q, K, V, mask=None):
        '''
        Parameters:
            Q - [b x l_q x (h * d_k)]
            K - [b x l_k x (h * d_k)]
            V - [b x l_k x (h * d_v)]
        '''
        nhead = self.nhead

        b_q, l_q, d_q = Q.shape
        b_k, l_k, d_k = K.shape
        b_v, l_v, d_v = V.shape

        assert b_q == b_k == b_v
        assert l_k == l_v
        assert d_q == d_k

        assert d_k % nhead == 0
        assert d_v % nhead == 0

        d_k = d_k // nhead
        d_v = d_v // nhead

        Q = self.W_Q(Q).view(b_q, l_q, nhead, d_k)
        K = self.W_K(K).view(b_q, l_k, nhead, d_k)
        V = self.W_V(V).view(b_q, l_k, nhead, d_v)

        Q = Q.transpose(1, 2).reshape(b_q * nhead, l_q, d_k)
        K = K.transpose(1, 2).reshape(b_q * nhead, l_k, d_k)
        V = V.transpose(1, 2).reshape(b_q * nhead, l_k, d_v)

        if mask is not None:
            b_m, d_1, d_2 = mask.shape
            assert b_m == b_q
            mask = mask.unsqueeze(1).expand(b_m, nhead, d_1, d_2).reshape(b_m * nhead, d_1, d_2)

        O = self.attention(Q, K, V, mask=mask)
        O = O.reshape(b_q, nhead, l_q, d_v).transpose(1, 2)
        O = O.reshape(b_q, l_q, nhead * d_v)
        O = self.dropout(self.W_O(O))
        return O
    pass


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1) -> None:
        super().__init__()
        self.W_1 = nn.Linear( input_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size,  input_size)
        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self, x):
        x = F.relu(self.W_1(x))
        x = self.W_2(x)
        x = self.dropout(x)
        return x
    pass


class TransformerEncoderLayer(nn.Module):
    def __init__(self, nhead, d_model, f_exp=4, dropout=0.1, d_k=None, d_v=None) -> None:
        super().__init__()
        self.attention   = MultiHeadAttention(nhead, d_model, dropout, d_k, d_v, bias=False)
        self.feedforward = PositionwiseFeedForward(d_model, d_model * f_exp)
        self.layernorm   = nn.LayerNorm(d_model, eps=1e-5)
        pass

    def forward(self, x, mask=None):
        r = x
        x = self.attention(x, x, x, mask=mask)

        x = x + r
        x = self.layernorm(x)

        r = x
        x = self.feedforward(x)

        x = x + r
        x = self.layernorm(x)
        return x
    pass


class TransformerDecoderLayer(nn.Module):
    def __init__(self, nhead, d_model, f_exp=4, dropout=0.1, d_k=None, d_v=None) -> None:
        super().__init__()
        self.masked_attention  = MultiHeadAttention(nhead, d_model, dropout, d_k, d_v, bias=False)
        self.enc_dec_attention = MultiHeadAttention(nhead, d_model, dropout, d_k, d_v, bias=False)
        self.feedforward       = PositionwiseFeedForward(d_model, d_model * f_exp, dropout)
        self.layernorm         = nn.LayerNorm(d_model, eps=1e-5)
        pass

    def forward(self, x, e, src_mask=None, tgt_mask=None):
        r = x
        x = self.masked_attention(x, x, x, mask=tgt_mask)

        x = x + r
        x = self.layernorm(x)

        r = x
        x = self.enc_dec_attention(x, e, e, mask=src_mask)

        x = x + r
        x = self.layernorm(x)

        r = x
        x = self.feedforward(x)

        x = x + r
        x = self.layernorm(x)
        return x
    pass


class TransformerUtility(object):
    def __init__(self) -> None:
        super().__init__()
        pass

    @staticmethod
    def positional_encoding(max_len, d_model):
        position   = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        multiplier = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        angle      = position * multiplier

        encoding          = torch.empty(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angle)
        encoding[:, 1::2] = torch.cos(angle[:, :d_model//2])
        return encoding

    @staticmethod
    def subsequent_mask(seq_len):
        mask = torch.ones((seq_len, seq_len))
        mask = torch.triu(mask, diagonal=1)
        mask = (1 - mask) == 0
        return mask
    pass


class TransformerEncoder(nn.Module):
    def __init__(self, nhead, d_model, nlayer, vocab_size=5000, max_seq_len=200, f_exp=4, dropout=0.1, d_k=None, d_v=None) -> None:
        super().__init__()
        self.register_buffer(
            name='positional_encoding',
            tensor=nn.Parameter(TransformerUtility.positional_encoding(max_seq_len, d_model), requires_grad=False)
        )

        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout        = nn.Dropout(dropout)
        self.encoder_stack  = nn.ModuleList([
            TransformerEncoderLayer(nhead, d_model, f_exp, dropout, d_k, d_v)
            for _ in range(nlayer)
        ])
        pass

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        x = self.word_embedding(x) + self.positional_encoding[:seq_len, :]
        x = self.dropout(x)

        for encoder in self.encoder_stack:
            x = encoder(x, mask=mask)
        return x
    pass


class TransformerDecoder(nn.Module):
    def __init__(self, nhead, d_model, nlayer, vocab_size=5000, max_seq_len=200, f_exp=4, dropout=0.1, d_k=None, d_v=None) -> None:
        super().__init__()
        self.register_buffer(
            name='positional_encoding',
            tensor=nn.Parameter(TransformerUtility.positional_encoding(max_seq_len, d_model), requires_grad=False)
        )

        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout        = nn.Dropout(dropout)
        self.decoder_stack  = nn.ModuleList([
            TransformerDecoderLayer(nhead, d_model, f_exp, dropout, d_k, d_v)
            for _ in range(nlayer)
        ])
        pass

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        batch_size, seq_len = x.shape

        x = self.word_embedding(x) + self.positional_encoding[:seq_len, :]
        x = self.dropout(x)

        for decoder in self.decoder_stack:
            x = decoder(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return x
    pass


class Transformer(nn.Module):
    def __init__(self,
            nhead, d_model, encoder_layers, decoder_layers, src_pad_idx, tgt_pad_idx,
            src_vocab_size=5000, tgt_vocab_size=5000, src_max_seq_len=200, tgt_max_seq_len=200,
            f_exp=4, dropout=0.1, d_k=None, d_v=None) -> None:
        super().__init__()
        self.encoder     = TransformerEncoder(nhead, d_model, encoder_layers, src_vocab_size, src_max_seq_len, f_exp, dropout, d_k, d_v)
        self.decoder     = TransformerDecoder(nhead, d_model, decoder_layers, tgt_vocab_size, tgt_max_seq_len, f_exp, dropout, d_k, d_v)
        self.linear      = nn.Linear(d_model, tgt_vocab_size)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        pass

    def forward(self, src, tgt, memory=None, src_mask=None, tgt_mask=None, device=None, return_dict=False):
        src_batch, src_seq_len = src.shape
        tgt_batch, tgt_seq_len = tgt.shape

        assert src_batch == tgt_batch

        if device is None:
            device = torch.device('cpu')

        if src_mask is None:
            src_mask = (src == self.src_pad_idx).unsqueeze(1)
        if tgt_mask is None:
            tgt_mask = (tgt == self.tgt_pad_idx).unsqueeze(1) | TransformerUtility.subsequent_mask(tgt_seq_len).to(device)

        if memory is None:
            memory = self.encoder(src, mask=src_mask)

        output = self.decoder(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.linear(output)

        if return_dict:
            return {
                'memory'  : memory,
                'logits'  : logits,
                'src_mask': src_mask,
                'tgt_mask': tgt_mask,
            }
        return logits
    pass

