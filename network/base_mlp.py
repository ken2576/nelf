import torch
import numpy as np


class PosEncoder(torch.nn.Module):
    def __init__(self, freq_num, freq_factor=np.pi):
        super().__init__()
        self.freq_num = freq_num
        self.freq_factor = freq_factor

    def forward(self, x):
        freq_multiplier = (
            self.freq_factor * 2 ** torch.arange(self.freq_num, device=x.device)
        )
        x_expand = x.unsqueeze(-1)
        sin_val = torch.sin(x_expand * freq_multiplier)
        cos_val = torch.cos(x_expand * freq_multiplier)
        return torch.cat(
            [x_expand, sin_val, cos_val], -1
        ).view(*x.shape[:-1], -1)


class MLP(torch.nn.Module):
    def __init__(self, feature_nums):
        super().__init__()

        self.input_size = feature_nums[0]
        self.output_size = feature_nums[-1]

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(feature_nums[0], feature_nums[1]))
        for i in range(1, len(feature_nums)-1):
            self.layers.append(torch.nn.PReLU(feature_nums[i]))
            self.layers.append(torch.nn.Linear(feature_nums[i], feature_nums[i+1]))

    def forward(self, x):
        input_shape = x.shape
        x = x.view(-1, self.input_size)
        for l in self.layers:
            x = l(x)
        return x.view(*input_shape[:-1], self.output_size)


class ResBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.prelu_0 = torch.nn.PReLU(input_size)
        self.fc_0 = torch.nn.Linear(input_size, hidden_size)
        self.prelu_1 = torch.nn.PReLU(hidden_size)
        self.fc_1 = torch.nn.Linear(hidden_size, output_size)

        self.shortcut = (
            torch.nn.Linear(input_size, output_size, bias=False)
            if input_size != output_size else None)
            

    def forward(self, x):
        residual = self.fc_1(self.prelu_1(self.fc_0(self.prelu_0(x))))
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return residual + shortcut



class ResBlocks(torch.nn.Module):
    def __init__(self, input_size, hidden_size, block_num, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.blocks = torch.nn.ModuleList([
            ResBlock(hidden_size, hidden_size, hidden_size if i < block_num - 1 else output_size)
            for i in range(block_num)
        ])
    
    def forward(self, x):
        input_shape = x.shape
        x = self.input_layer(x.view(-1, self.input_size))
        for block in self.blocks:
            x = block(x)
        return x.view(*input_shape[:-1], self.output_size)



class PosEncodeResnet(torch.nn.Module):
    def __init__(
        self,
        posencode_size, nonencode_size, hidden_size, output_size,
        posencode_freq_num, block_num
    ):
        super().__init__()
        
        self.input_size = (
            posencode_size * (2 * posencode_freq_num + 1)
            + nonencode_size
        )
        self.output_size = output_size

        if posencode_size > 0:
            self.pos_encoder = PosEncoder(posencode_freq_num)

        self.input_layer = torch.nn.Linear(self.input_size, hidden_size)
        self.blocks = torch.nn.ModuleList(
            [ResnetBlock(hidden_size, hidden_size, hidden_size)
             for i in range(block_num)]
        )
        self.output_prelu = torch.nn.PReLU(hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)


    def forward(self, posencode_x, nonencode_x):
        x = []
        if posencode_x is not None:
            x.append(self.pos_encoder(posencode_x))
        if nonencode_x is not None:
            x.append(nonencode_x)
        x = torch.cat(x, axis=-1)

        input_shape = x.shape
        x = self.input_layer(x.view(-1, self.input_size))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(self.output_prelu(x)).view(*input_shape[:-1], self.output_size)
