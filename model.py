from typing import Dict

import torch
from torch.nn import Embedding

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.gru = torch.nn.GRU(len(embeddings[0]), hidden_size, num_layers,
                                bidirectional=bidirectional, dropout=dropout)
        self.linear = torch.nn.Linear(self.encoder_output_size, num_class)
        # COMPLETE

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional == True:
            return self.hidden_size * 2
        else:
            return self.hidden_size
        # COMPLETE

    def forward(self, batch) -> Dict[str, torch.Tensor]: # -> ????
        # TODO: implement model forward
        batch_trans = self.embed(batch.t())
        output, _ = self.gru(batch_trans)
        output = self.linear(output[-1])
        return output
        # COMPLETE

class SeqSlot(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqSlot, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.embed = Embedding.from_pretrained(embeddings, freeze=True)
        # TODO: model architecture
        self.gru = torch.nn.GRU(len(embeddings[0]), hidden_size, num_layers,
                                bidirectional=bidirectional, dropout=dropout)
        
        self.linear = torch.nn.Linear(self.encoder_output_size, num_class)
        self.dropout = torch.nn.Dropout(dropout)
        # COMPLETE

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional == True:
            return self.hidden_size * 2
        else:
            return self.hidden_size
        # COMPLETE

    def forward(self, batch, lens) -> Dict[str, torch.Tensor]: # -> ????
        # TODO: implement model forward
        embeded = self.embed(batch.t())
        pack = pack_padded_sequence(embeded, lens)
        output, _ = self.gru(pack)
        output, _ = pad_packed_sequence(output) 
        output = self.linear(output)
        return output
        # COMPLETE

