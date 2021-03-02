#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict


NUM_FEEDBACK_BITS = 420  # pytorch版本一定要有这个参数

#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Defining Function Defining
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)
    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2
    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)

def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num
#=======================================================================================================================
#=======================================================================================================================
# Quantization and Dequantization Layers Defining
class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None
class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out
class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out
#=======================================================================================================================
#=======================================================================================================================
# Encoder and Decoder Class Defining
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class ConvBNNoPadding(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        super(ConvBNNoPadding, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=0, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class DeConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, padding=[0, 0], output_padding=0, stride=1, groups=1):
        super(DeConvBN, self).__init__(OrderedDict([
            ('deconv', nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, output_padding=output_padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class DeConvBNPadding(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, padding=1, output_padding=1, stride=1, groups=1):
        super(DeConvBNPadding, self).__init__(OrderedDict([
            ('deconv', nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, output_padding=output_padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class ResBlock(nn.Module):
    def __init__(self, di, dh):
        super(ResBlock, self).__init__()
        self.BN1 = nn.BatchNorm2d(dh)
        self.BN2 = nn.BatchNorm2d(di)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(di, dh, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(dh, di, 3, padding=1, stride=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.relu2(x + self.BN2(self.conv2(self.relu1(self.BN1(self.conv1(x))))))

class BottleneckResBlock(nn.Module):
    def __init__(self, di, dh):
        super(BottleneckResBlock, self).__init__()
        self.BN1 = nn.BatchNorm2d(dh)
        self.BN2 = nn.BatchNorm2d(dh)
        self.BN3 = nn.BatchNorm2d(di)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.conv1 = nn.Conv2d(di, dh, 1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(dh, dh, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(dh, di, 1, padding=0, stride=1)


    def forward(self, x):
        return x + self.BN3(self.conv3(self.BN2(self.conv2(self.relu1(self.BN1(self.conv1(x)))))))

class Passthroughlayer(nn.Module):
    def __init__(self, h, w, c):
        super(Passthroughlayer, self).__init__()
        self.h = h
        self.w = w
        self.c = c
    def forward(self, x):
        # x_down = torch.cat((x[:, :, 0:self.h//2, 0:self.w//2], x[:, :, 0:self.h//2, self.w//2:]), dim=1)
        # x_down = torch.cat((x_down, x[:, :, self.h//2:, 0:self.w//2]), dim=1)
        # x_down = torch.cat((x_down, x[:, :, self.h//2:, self.w//2:]), dim=1)
        x_down = torch.cat((x[:, :, 0:self.h:2, 0:self.w:2], x[:, :, 1:self.h:2, 0:self.w:2]), dim=1)
        x_down = torch.cat((x_down, x[:, :, 0:self.h:2, 1:self.w:2]), dim=1)
        x_down = torch.cat((x_down, x[:, :, 1:self.h:2, 1:self.w:2]), dim=1)
        return x_down

class DePassthroughlayer(nn.Module):
    def __init__(self, h, w, c):
        super(DePassthroughlayer, self).__init__()
        self.h = h
        self.w = w
        self.c = c
    def forward(self, x):
        # x_down = torch.cat((x[:, :, 0:self.h//2, 0:self.w//2], x[:, :, 0:self.h//2, self.w//2:]), dim=1)
        # x_down = torch.cat((x_down, x[:, :, self.h//2:, 0:self.w//2]), dim=1)
        # x_down = torch.cat((x_down, x[:, :, self.h//2:, self.w//2:]), dim=1)
        x_down = torch.cat((x[:, 0, :, :], x[:, 1, :, :]), dim=1)
        x_down_temp = torch.cat((x[:, 2, :, :], x[:, 3, :, :]), dim=1)
        # print(x_down.shape)
        # print(x_down_temp.shape)
        x_down = torch.cat((x_down, x_down_temp), dim=2)
        x_down = x_down.view(-1, 1, 16, 20)
        return x_down


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(256, 256, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('conv1x9', ConvBN(256, 256, [1, 7])),
            ('relu2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('conv9x1', ConvBN(256, 256, [7, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(256, 256, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('conv5x1', ConvBN(256, 256, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(256*2, 256, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
            super(SEBottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.se = SELayer(planes, reduction)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class Encoder(nn.Module):
    num_quan_bits = 3
    def __init__(self, feedback_bits, quantization=True):
        num_quan_bits = 3
        super(Encoder, self).__init__()
        self.conv_input = ConvBN(32, 128, 1)
        # self.pad = nn.ZeroPad2d(padding=(2, 2, 0, 0))
        self.relu1 = nn.LeakyReLU(0.2)
        self.passthrough = Passthroughlayer(16, 20, 2)        # 8， 10
        self.passthrough2 = Passthroughlayer(8, 10, 8)   # 4, 5
        # self.CRBlock = CRBlock()
        self.res1 = BottleneckResBlock(128, 128*6)
        self.res2 = BottleneckResBlock(128, 128*6)
        self.res3 = ResBlock(128, 256)
        self.res4 = ResBlock(128, 256)
        self.res5 = ResBlock(128, 256)
        self.res6 = ResBlock(128, 256)
        # self.SEBlock = SEBottleneck(256, 256)
        self.conv_output_1 = ConvBN(128, 32, 3)
        self.relu2 = nn.ReLU()
        self.conv_output_2 = nn.Conv2d(32, 7, 1)
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(num_quan_bits)
        self.quantization = quantization

    def forward(self, x):
        x = x.transpose(1, 3)
        x = torch.cat((x[:, :, :, 20:24], x[:, :, :, 0:16]), dim=3)  # batch, 2, 16, 20

        out = self.passthrough(x)
        out = self.passthrough2(out)
        out = self.relu1(self.conv_input(out))
        # out = self.CRBlock(out)
        out = self.res3(self.res2(self.res1(out)))
        out = self.res6(self.res5(self.res4(out)))
        out = self.conv_output_2(self.relu2(self.conv_output_1(out)))  # batch, 4, 16, 2
        out = out.view(-1, 4*5*7)
        out = self.sig(out)

        if self.quantization:
            out = self.quantize(out)
        else:
            out = out
        return out

class Decoder(nn.Module):
    num_quan_bits = 3

    def __init__(self, feedback_bits, quantization=True):
        num_quan_bits = 3
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(num_quan_bits)
        # self.conv_input = nn.ConvTranspose2d(2, 128, 4, stride=2, padding=[1, 1])
        self.conv_input = ConvBN(7, 128*16, 3)
        self.px = nn.PixelShuffle(4)
        # self.conv_input2 = ConvBN(1, 200, 3)
        self.relu1 = nn.LeakyReLU(0.2)

        self.CRB1 = ResBlock(128, 256)
        self.CRB2 = ResBlock(128, 256)
        self.CRB3 = ResBlock(128, 256)
        self.CRB4 = ResBlock(128, 256)
        self.CRB5 = ResBlock(128, 256)
        self.CRB6 = ResBlock(128, 256)
        self.out_cov = nn.Conv2d(128, 2, 1)
        self.sig = nn.Sigmoid()
        self.quantization = quantization

    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = out.view(-1, 7, 4, 5)
        # out = self.pad(out)
        out = self.px(self.conv_input(out))  # 4, 8, 10
        # out = self.depass(out)
        # out = self.relu1(self.conv_input2(out))
        out = self.CRB3(self.CRB2(self.CRB1(out)))
        out = self.CRB6(self.CRB5(self.CRB4(out)))
        out = self.out_cov(out)
        out = self.sig(out)

        batch_size = len(out)
        pad = torch.ones([batch_size, 2, 16, 4]) * 0.5
        pad = pad.cuda()
        out1 = out[:, :, :, 0:4]
        out = torch.cat((out[:, :, :, 4:20], pad), dim=3)
        out = torch.cat((out, out1), dim=3)
        out = out.transpose(1, 3)
        # print(out.shape)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)
    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out



#=======================================================================================================================
#=======================================================================================================================
# NMSE Function Defining
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def Score(NMSE):
    score = 1 - NMSE
    return score


def NMSE_cuda(x, x_hat):
    x_real = x[:, :, :, 0].view(len(x), -1)
    x_imag = x[:, :, :, 1].view(len(x), -1)
    x_hat_real = x_hat[:, :, :, 0].view(len(x_hat), -1)
    x_hat_imag = x_hat[:, :, :, 1].view(len(x_hat), -1)

    power = torch.sum((x_real-0.5) ** 2 + (x_imag-0.5) ** 2, axis=1)
    mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


#=======================================================================================================================
#=======================================================================================================================
# Data Loader Class Defining
class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]




