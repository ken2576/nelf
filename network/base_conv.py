import torch


class NormAct(torch.nn.Module):
    def __init__(self, channel_num, act='prelu'):
        super().__init__()

        self.norm = torch.nn.GroupNorm(min(32, channel_num), channel_num)
        if act == 'prelu':
            self.act = torch.nn.PReLU(channel_num)
        elif act == 'softplus':
            self.act = torch.nn.Softplus()
        elif act == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return self.act(self.norm(x))



class NormActConv(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride):
        super().__init__()

        self.norm = torch.nn.GroupNorm(min(32, input_channel), input_channel)
        self.act = torch.nn.PReLU(input_channel)
        self.conv = torch.nn.Conv2d(
            input_channel, output_channel,
            kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        return self.conv(self.act(self.norm(x)))
        x = self.act(self.norm(x))
        i, j = torch.meshgrid(
            torch.arange(x.shape[-2], device=x.device), 
            torch.arange(x.shape[-1], device=x.device),
        )
        x = torch.cat([
            x,
            ((i - x.shape[-2] / 2) / x.shape[-2]).expand(1, 1, -1, -1),
            ((j - x.shape[-1] / 2) / x.shape[-1]).expand(1, 1, -1, -1),
        ], axis=1)
        return self.conv(x)



class ConvNormAct(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, act='prelu'):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            input_channel, output_channel,
            kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2
        )
        self.norm = torch.nn.GroupNorm(min(32, output_channel), output_channel)
        if act == 'prelu':
            self.act = torch.nn.PReLU(output_channel)
        elif act == 'softplus':
            self.act = torch.nn.Softplus()
        elif act == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))



class ResConvBlock(torch.nn.Module):
    def __init__(self, input_channel, hidden_channel, output_channel, stride):
        super().__init__()

        self.normactconv0 = NormActConv(input_channel, hidden_channel, 3, stride)
        self.normactconv1 = NormActConv(hidden_channel, output_channel, 3, 1)

        if stride == 1 and input_channel == output_channel:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Conv2d(
                input_channel, output_channel,
                kernel_size=1, stride=stride, padding=0
            )

    def forward(self, x):
        return self.shortcut(x) + self.normactconv1(self.normactconv0(x))
