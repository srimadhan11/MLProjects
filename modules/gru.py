import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderGRU(nn.Module):
    def __init__(self, input_vocab_size, embedding_size, hidden_size, num_layers=2, bidirectional=True, dropout_p=0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first=False)
        pass

    def forward(self, input_sequence, pad_idx):
        '''
        Parameter
            input_sequence: LxB
            pad_idx: 1 (an integer)
        Return
            output: LxBx(D*H)
            hidden: (D*N)xBxH

        where,
            * B - batch size
            * L - sequence length
            * H - encoder hidden size
            * M - embedding hidden size
            * D - number of directions (2 if bidirectional, else 1)
            * N - number of layers

        '''
        embedded = self.embedding(input_sequence)   # LxBxM

        lengths_without_pad = (input_sequence.size(0) - (input_sequence == pad_idx).sum(0)).cpu()
        pack_padded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths_without_pad, batch_first=False, enforce_sorted=False)
        pack_padded, hidden = self.gru(pack_padded)
        output, pad_lengths = torch.nn.utils.rnn.pad_packed_sequence(pack_padded, batch_first=False, total_length=input_sequence.size(0))

        assert (lengths_without_pad == pad_lengths).all()

        return output, hidden                       # LxBx(D*H), (D*N)xBxH
    pass


class DecoderGRU(nn.Module):
    def __init__(self, output_vocab_size, embedding_size, hidden_size, num_layers=2, dropout_p=0.2):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout_p, bidirectional=False, batch_first=False)
        pass

    def forward(self, output_sequence, hidden, pad_idx):
        '''
        Parameter
            output_sequence: LxB
            hidden: NxBxH
            pad_idx: 1 (an integer)
        Return
            output: LxBxH
            hidden: NxBxH

        where,
            * B - batch size
            * L - sequence length
            * H - decoder hidden size
            * M - embedding hidden size
            * N - number of layers
        '''
        embedded = self.embedding(output_sequence)      # LxBxM

        lengths_without_pad = (output_sequence.size(0) - (output_sequence == pad_idx).sum(0)).cpu()
        pack_padded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths_without_pad, batch_first=False, enforce_sorted=False)
        pack_padded, hidden = self.gru(pack_padded, hidden)
        output, pad_lengths = torch.nn.utils.rnn.pad_packed_sequence(pack_padded, batch_first=False, total_length=output_sequence.size(0))

        assert (lengths_without_pad == pad_lengths).all()

        return output, hidden                           # LxBxH, NxBxH
    pass


class AttentionDecoderGRU(nn.Module):
    def __init__(self, output_vocab_size, embedding_size, hidden_size, num_layers=2, dropout_p=0.2):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_size)
        self.attn = BahdanauAttention(hidden_size, hidden_size)
        self.gru  = nn.GRU(embedding_size+hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_p, bidirectional=False, batch_first=False)
        pass

    def forward(self, output_sequence, encoder_hiddens, hidden):
        '''
        Parameter
            output_sequence: B
            encoder_hiddens: LxBxH
            hidden         : NxBxH
        Return
            output: 1xBxH
            hidden: NxBxH

        where,
            * B - batch size
            * L - sequence length
            * H - decoder hidden size
            * N - number of layers
            * M - embedding hidden size
        '''
        embedded        = self.embedding(output_sequence)           # BxM
        encoder_hiddens = encoder_hiddens.transpose(0, 1)           # BxLxH
        attention       = self.attn(encoder_hiddens, hidden[-1])    # BxH

        combined = torch.cat((embedded, attention), dim=1)          # Bx(M+H)
        combined = combined.unsqueeze(0)                            # 1xBx(M+H)

        output, hidden = self.gru(combined, hidden)                 # 1xBxH, NxBxH
        return output, hidden
    pass


class EncoderDecoderGRU(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size, hidden_size,
                 src_pad_idx, tgt_pad_idx, num_layers=2, bidirectional=True, dropout_p=0.2):
        super().__init__()
        encoder_hidden_size = hidden_size
        decoder_hidden_size = hidden_size * (2 if bidirectional else 1)

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = EncoderGRU(input_vocab_size , embedding_size, encoder_hidden_size, num_layers, bidirectional, dropout_p)
        self.decoder = DecoderGRU(output_vocab_size, embedding_size, decoder_hidden_size, num_layers, dropout_p)
        self.linear  = nn.Linear(decoder_hidden_size, output_vocab_size)
        pass

    def forward(self, src, tgt, encoder_output=None, encoder_hidden=None, return_dict=False):
        Bs, Ls = src.shape
        Bt, Lt = tgt.shape
        assert Bs == Bt

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        if None in (encoder_output, encoder_hidden):
            encoder_output, encoder_hidden = self.encoder(src, self.src_pad_idx)
            if self.bidirectional:
                encoder_hidden = encoder_hidden.reshape(self.num_layers, 2, Bs, -1).transpose(1, 2)
                encoder_hidden = encoder_hidden.reshape(self.num_layers   , Bs, -1)

        decoder_output, decoder_hidden = self.decoder(tgt, encoder_hidden, self.tgt_pad_idx)

        logits = self.linear(decoder_output)
        logits = logits.transpose(0, 1)

        if return_dict:
            return {
                'encoder_output': encoder_output,
                'encoder_hidden': encoder_hidden,
                'logits'        : logits,
            }
        return logits
    pass


class AttentionEncoderDecoderGRU(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size, hidden_size,
                 src_pad_idx, tgt_pad_idx, num_layers=2, bidirectional=True, dropout_p=0.2):
        super().__init__()
        encoder_hidden_size = hidden_size
        decoder_hidden_size = hidden_size * (2 if bidirectional else 1)

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = EncoderGRU(input_vocab_size, embedding_size, encoder_hidden_size, num_layers, bidirectional, dropout_p)
        self.decoder = AttentionDecoderGRU(output_vocab_size, embedding_size, decoder_hidden_size, num_layers, dropout_p)
        self.linear  = nn.Linear(decoder_hidden_size, output_vocab_size)
        pass

    def forward(self, src, tgt, encoder_output=None, encoder_hidden=None, return_dict=False):
        Bs, Ls = src.shape
        Bt, Lt = tgt.shape
        assert Bs == Bt

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        if None in (encoder_output, encoder_hidden):
            encoder_output, encoder_hidden = self.encoder(src, self.src_pad_idx)
            if self.bidirectional:
                encoder_hidden = encoder_hidden.reshape(self.num_layers, 2, Bs, -1).transpose(1, 2)
                encoder_hidden = encoder_hidden.reshape(self.num_layers   , Bs, -1)

        decoder_hidden = encoder_hidden.clone()
        decoder_output = torch.zeros((Lt, Bt, encoder_hidden.shape[2]), device=src.device, dtype=torch.float32)
        for l in range(Lt):
            mask = ~(tgt[l] == self.tgt_pad_idx)
            if not mask.any():
                break
            decoder_output[l, mask], decoder_hidden[:, mask] = self.decoder(tgt[l, mask], encoder_output[:, mask], decoder_hidden[:, mask])

        logits = self.linear(decoder_output)
        logits = logits.transpose(0, 1)

        if return_dict:
            return {
                'encoder_output': encoder_output,
                'encoder_hidden': encoder_hidden,
                'logits'        : logits,
            }
        return logits
    pass

