from torch.nn import Module
import torch
import torch.nn as nn
from typing import List, Dict

class PositionalEncoding(Module):

    def __init__(self, device= torch.device('cpu')):
        super(PositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position_encoding = torch.Tensor(
            [[pos / torch.pow(10000, 2.0 * (j // 2) / x.size()[1])
              for j in torch.arange(0, x.size()[1])]
             for pos in torch.arange(0, x.size()[0])])
        position_encoding[:, 0::2] = torch.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = torch.cos(position_encoding[:, 1::2])

        return x + position_encoding

class ScaledDotProductAttention(Module):

    def __init__(self, input_size:List, attention_dropout= 0.0, Sequence:bool= False, device= torch.device('cpu')):
        super(ScaledDotProductAttention, self).__init__()
        
        self.input_size = input_size
        self.Sequence = Sequence

        #运行需要函数
        self.softmax_layer = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(attention_dropout)
        self.device = device
        self.d = torch.Tensor([self.input_size[2]]).sqrt().to(device)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor= None) -> torch.Tensor:

        Z = torch.div(torch.bmm(Q, K.permute(0, 2, 1)), self.d)

        if mask is not None:
            Z.masked_fill(mask == 0, -1e9)
        if self.Sequence == True:
            sequence_mask = torch.ones([Z.size(1), Z.size(2)]).to(self.device)
            Z.masked_fill(sequence_mask.triu(0).t() == 0, -1e9)

        attention = self.softmax_layer(Z)
        Z = torch.bmm(self.dropout(attention), V)
        return Z, attention

class MultiHeadAttention(Module):

    def __init__(self, input_size:List, attention_dropout= 0.0, num_heads:int= 8, hidden_dim:int= 64, Sequence:bool= False, device= torch.device('cpu')):
        """

        :param input_size: 输入维度
        :param hidden_dim:
        :param Sequence:
        """
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.Sequence = Sequence
        self.device = device

        self.linear_q = nn.Linear(in_features= self.input_size[2], out_features= self.hidden_dim * num_heads, bias= False)
        self.linear_k = nn.Linear(in_features=self.input_size[2], out_features=self.hidden_dim * num_heads, bias=False)
        self.linear_v = nn.Linear(in_features=self.input_size[2], out_features=self.hidden_dim * num_heads, bias=False)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.input_size, attention_dropout, self.Sequence, device= self.device)
        self.linear_output = nn.Linear(in_features= self.num_heads * self.hidden_dim, out_features=self.input_size[2])
        self.layer_norm = nn.LayerNorm(self.input_size[-1])
        self.final_linear = nn.Sequential(
            nn.Linear(in_features=self.input_size[2], out_features=2048),
            nn.Dropout(attention_dropout),
            nn.Linear(in_features=2048, out_features=self.input_size[2])
        )


    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor= None) -> torch.Tensor:

        X = V
        Q = self.linear_q(Q).view(self.input_size[0] * self.num_heads, -1, self.hidden_dim)
        K = self.linear_k(K).view(self.input_size[0] * self.num_heads, -1, self.hidden_dim)
        V = self.linear_v(V).view(self.input_size[0] * self.num_heads, -1, self.hidden_dim)

        mask = mask.repeat([self.num_heads, 1, 1])
        Z, attention = self.scaled_dot_product_attention(Q, K, V, mask)
        Z = self.linear_output(Z.view(self.input_size[0], -1, self.num_heads * self.hidden_dim))

        Z = self.layer_norm(X + Z)
        X = Z

        return self.layer_norm(X + self.final_linear(Z))

class Feed_Forward(Module):

    def __init__(self, input_size: List, hidden_dim:int= 2048, dropout= 0.0, device= torch.device('cpu')):

        super(Feed_Forward, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channels= input_size[2], out_channels= hidden_dim, kernel_size= 1),
            nn.ReLU(),
            nn.Conv1d(in_channels= hidden_dim, out_channels= input_size[2], kernel_size= 1),
        )
        self.layer_norm = nn.LayerNorm(input_size[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.feed_forward(x.permute(0, 2, 1))
        y = self.dropout(y).permute(0, 2, 1)

        return self.layer_norm(x + y)

class Encoder_layer(Module):

    def __init__(self, input_size, dropout, device= torch.device('cpu')):
        super(Encoder_layer, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        self.device = device
        self.attention = MultiHeadAttention(input_size, attention_dropout= dropout, Sequence=False, device= self.device)
        self.feed_forward = Feed_Forward(input_size, dropout=dropout, device= self.device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.attention(x, x, x, mask)
        return self.feed_forward(x)

class Encoder(Module):
    def __init__(self, input_size, num_layers= 6, dropout= 0.0, device= torch.device('cpu')):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.encoder_layer = nn.ModuleList([Encoder_layer(input_size, dropout, device= self.device) for i in torch.arange(0, num_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        for encoder in self.encoder_layer:
            x = encoder(x, mask)
        return x

class Decoder_layer(Module):

    def __init__(self, input_size, dropout= 0.0, device= torch.device('cpu')):
        super(Decoder_layer, self).__init__()
        self.input_size = input_size
        self.device = device
        self.attention_Mask = MultiHeadAttention(input_size, attention_dropout=dropout, Sequence=True, device= self.device)
        self.attention = MultiHeadAttention(input_size, attention_dropout=dropout, Sequence=False, device= self.device)
        self.feed_forward = Feed_Forward(input_size, dropout=dropout, device= self.device)

    def forward(self, x: torch.Tensor, encoder_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.attention_Mask(x, x, x, mask)
        x = self.attention(encoder_input, encoder_input, x, mask)

        return self.feed_forward(x)

class Decoder(Module):

    def __init__(self, input_size, num_layers= 6, dropout= 0.0, device= torch.device('cpu')):
        super(Decoder, self).__init__()
        self.device = device
        self.decoder_layer = nn.ModuleList([Decoder_layer(input_size, dropout, device=self.device) for _ in torch.arange(0, num_layers)])

    def forward(self, x: torch.Tensor, encoder_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for decoder in self.decoder_layer:
            x = decoder(x, encoder_input, mask)
        return x

class Transformer(Module):

    def __init__(self, input_size, vocab: Dict, dropout= 0.1, pre_hidden_dim= 2048,device= torch.device('cpu')):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.vocab = vocab
        self.dropout = dropout

        self.encoder = Encoder(self.input_size, dropout= self.dropout)
        self.decoder = Decoder(self.input_size, dropout= self.dropout)
        self.positional_encoding = PositionalEncoding(device)
        self.pred = nn.Sequential(
            nn.Linear(in_features= input_size[2], out_features= pre_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(in_features= pre_hidden_dim, out_features= len(list(vocab["text_word_to_indexer"].keys()))),
            nn.Softmax(dim= 1)
        )

    def forward(self, input: torch.Tensor,tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(self.positional_encoding(input), mask)
        output = self.decoder(tgt, output, mask)
        return output

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = torch.ones([4, 10, 20]).to(device)
    mask = torch.ones([4, 10, 10]).triu(0).permute([0, 2, 1]).to(device)
    encoder = Encoder([4, 10, 20], dropout= 0.1, device= device)
    encoder = encoder.cuda()
    output = encoder(input, mask)
    print(output.size())
