import numpy as np
import timm
import torch
import torch.nn.functional as F
import torch_dct as dct
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.utils import save_image
import cv2
import albumentations as A
import tensorly as tl
# from .PVTv1 import pvt_v1
from .backbone.Res2Net_v1b import res2net50_v1b_26w_4s
from .pvt_v2_eff import pvt_v2_eff_b4
# from .CFM import TRAM2,SpatialAttention
# from .CFM import Octave
from .utils.base_model import BasicModelClass
from .MSAModule import MSA_module

tl.set_backend('pytorch')

def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError

def _to_2tuple(x):
    """将输入转换为2元组"""
    if isinstance(x, int):
        return (x, x)
    else:
        return tuple(x)

# 卷积层-批归一化-激活函数
class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        super().__init__()
        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=_to_2tuple(stride),
                padding=_to_2tuple(padding),
                dilation=_to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name))

class two_ConvBnRule(nn.Module):
    def __init__(self, in_chan, out_chan=64):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)
        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)
        return feat

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        # def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# 特征通道数转换
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.aspp = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1, 0),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.aspp(x)

############### 频域特征提取模块 ###############
class DCTLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        original_shape = x.shape  # 对于输入 [4,64,12,12]，original_shape = (4,64,12,12)
        # batch_size, channels, H_s, W = original_shape  # 显式解析各维度，更清晰
        # print("dctlayer输入形状为：", x.shape)
        x = x.view(-1, *x.shape[-2:])   

        freq = dct.dct_2d(x, norm='ortho')

        # freq = freq.view(*x.shape[:-2], *x.shape[-2:])
        freq = freq.view(*original_shape)
    
        return freq

class SpectralAttention(nn.Module):
    """
    频谱注意力机制
    对DCT变换后的频域特征进行注意力加权
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算全局平均池化和最大池化的频域特征
        avg_out = self.mlp(self.global_avg_pool(x))
        max_out = self.mlp(self.global_max_pool(x))
        
        # 融合两种池化结果
        out = avg_out + max_out
        return x * self.sigmoid(out)

class EfficientFreqFusion(nn.Module):
    """
    高效的频域融合模块
    简洁设计，直接在频域操作
    """
    def __init__(self, in_dim, mm_size):
        super().__init__()
        self.dct_layer = DCTLayer()
        self.spectral_attention = SpectralAttention(in_dim)
        
        # 频域特征增强
        self.freq_enhance = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim)
        )
        
        # 差异提取和融合
        self.diff_fusion = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出调整
        self.output_adjust = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, o, a1, a2):
        # DCT变换
        o_freq = self.dct_layer(o)
        a1_freq = self.dct_layer(a1)
        a2_freq = self.dct_layer(a2)
        
        # 应用频谱注意力
        o_attn = self.spectral_attention(o_freq)
        a1_attn = self.spectral_attention(a1_freq)
        a2_attn = self.spectral_attention(a2_freq)
        
        # 增强频域特征
        o_enhanced = self.freq_enhance(o_attn)
        a1_enhanced = self.freq_enhance(a1_attn)
        a2_enhanced = self.freq_enhance(a2_attn)
        
        # 计算跨视角差异
        diff1 = torch.abs(o_enhanced - a1_enhanced)
        diff2 = torch.abs(o_enhanced - a2_enhanced)
        
        # 融合差异信息
        diff_fused = self.diff_fusion(torch.cat([diff1, diff2], dim=1))
        
        # 最终输出：原始增强特征 + 差异信息
        output = self.output_adjust(o_enhanced + diff_fused)
        
        return output

############### 多视角生成 ###############

class DataProcessor:
    def __init__(self):
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.base_shape = (384, 384)
        self.count = 0

    def process_batch_image(self, images):
        """处理批量图像，支持任意batch size（如8）"""
        # 存储所有样本的多视角结果

        images_a1 = torch.flip(images,dims=[2]) # 垂直翻转
        images_a2 = torch.flip(images,dims=[3]) # 水平翻转

        # save_image(images.float(), "debug_input_batch.png")  # 保存输入批次图像
        # save_image(images_a1.float(), "debug_input_batch_a1.png")  # 保存输入批次图像
        # save_image(images_a2.float(), "debug_input_batch_a2.png")  # 保存输入批次图像
        # while 1:
        #     ...
        
        # 拼接批次维度（将列表转换为 (batch_size, C, H, W) 的张量）
        return {
            "image_o": images,
            "image_a1": images_a1,
            "image_a2": images_a2
        }

############### 可变形卷积对齐模块 ###############
class DeformableAlign(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

############### 频域增强与视角融合模块 ###############
class CAMV(nn.Module):
    def __init__(self, in_dim, mm_size):
        super().__init__()
        self.dct_layer = DCTLayer()
        self.fuse_conv = ConvBNReLU(in_dim * 3, in_dim, 3, 1, 1)
        
        self.dcn = DeformableAlign(in_dim)
        
        self.attention_head = nn.Sequential(
            nn.Conv2d(in_dim * 3, 3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, o, a1, a2):
        o_low, o_high = self.dct_layer(o)
        a1_low, a1_high = self.dct_layer(a1)
        a2_low, a2_high = self.dct_layer(a2)

        o_high = F.interpolate(o_high, scale_factor=2, mode='bilinear', align_corners=False)
        a1_high = F.interpolate(a1_high, scale_factor=2, mode='bilinear', align_corners=False)
        a2_high = F.interpolate(a2_high, scale_factor=2, mode='bilinear', align_corners=False)
        # print("o_high形状：", o_high.shape) #6

        high_diff_o_a1 = torch.abs(o_high - a1_high)
        high_diff_o_a2 = torch.abs(o_high - a2_high)

        fused_freq = torch.cat([o_high, high_diff_o_a1, high_diff_o_a2], dim=1)
        fused_freq = self.fuse_conv(fused_freq)
        # print("fused_freq形状：", fused_freq.shape)
        
        o_aligned = self.dcn(o + fused_freq)
        a1_aligned = self.dcn(a1)
        a2_aligned = self.dcn(a2)
        
        concat = torch.cat([o_aligned, a1_aligned, a2_aligned], dim=1)
        attn = self.attention_head(concat)
        fused = o_aligned * attn[:,0:1] + a1_aligned * attn[:,1:2] + a2_aligned * attn[:,2:3]
        return fused # channel 64

class TransLayer(nn.Module):
    """特征通道统一层"""
    def __init__(self, out_c, last_module=ASPP):
        super().__init__()
        # self.c5_down = nn.Sequential(
        #     last_module(in_dim=2048, out_dim=out_c),
        # )
        # self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, padding=1))
        # self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, padding=1))
        # self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, padding=1))
        # self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, padding=1))

        # self.c5_down = two_ConvBnRule(2048, out_c)
        self.c4_down = two_ConvBnRule(512, out_c)
        self.c3_down = two_ConvBnRule(320, out_c)
        self.c2_down = two_ConvBnRule(128, out_c)
        self.c1_down = two_ConvBnRule(64, out_c)     

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 4
        c1, c2, c3, c4 = xs
        # print("###",c1.shape)
        # print("###",c2.shape)
        # print("###",c3.shape)
        # while 1:
        #     ...
        # c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c1, c2, c3, c4

class CFU(nn.Module):
    """Cross Feature Fusion Unit"""
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups
        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1) # 通道倍增再分为四个分支
        outs = []
        branch_out = self.interact["0"](xs[0]) # 第一个分支交互，输出2倍hidden_dim的特征
        outs.append(branch_out.chunk(2, dim=1)) # 分成两部分，一部分作为当前分支输出，一部分作为下一分支输入 每部分hidden_dim维度
        # outs[0]为tuple

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(2, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(1, dim=1))
        out = torch.cat([o[0] for o in outs], dim=1)
        out = self.fuse(out)
        return self.final_relu(out + x)

# 解码器
class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 放大两倍
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):  # 深层次特征->浅层特征 f4,f3,f2
        x1_1 = x1
        # f4nc (64*22*22)
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2  # (64*22*22)*(64*22*22)=(64*22*22)
        # f3nc (64*44*44)
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3  # (64*44*44)
        # concat f4nc和f5nc
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)  # (128*22*22)
        x2_2 = self.conv_concat2(x2_2)  # (128*22*22)
        # concat f3nc 和 conv(f4nc,f5nc)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)  # (192*44*44)
        x3_2 = self.conv_concat3(x3_2)  # (192*44*44)

        x = self.conv4(x3_2)
        x = self.conv5(x)  # (1*44*44)

        return x

# 特征提取模块
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    
# Group-Reversal Attention (GRA) Block
class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y

class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y

class MyEncoderResNet(nn.Module):
    def __init__(self, imagenet_pretrained=True):
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 12, 12
        return x1, x2, x3, x4

class MyEncoderPvt2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pvt = pvt_v2_eff_b4(pretrained=False) # 使用PVTv2-B4作为特征提取器，加载ImageNet预训练权重
        # 加载本地预训练权重data\pretrain\pvt_v2_b4.pth
        local_weight_path = "./data/pretrain/pvt_v2_b4.pth"  # 本地权重路径
        state_dicts = torch.load(local_weight_path, map_location='cuda')  # 加载权重
        state_dicts.pop("head.weight")
        state_dicts.pop("head.bias")
        # # 适配特征提取器的权重（移除分类头相关参数，因features_only=True时模型无分类头）
        # # 过滤掉state_dict中与shared_encoder结构不匹配的键（如分类头的fc层参数）
        # filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.shared_encoder.state_dict()}
        # 加载过滤后的权重
        self.pvt.load_state_dict(state_dicts)

    def forward(self, x):
        x = self.pvt(x)  
        return x["reduction_2"], x["reduction_3"], x["reduction_4"], x["reduction_5"]  # 返回四个阶段的特征

############### 主模型DCTNet ###############
class DCTNet(BasicModelClass):
    def __init__(self, channel=128):
        super().__init__()
        # # 特征提取器 - 不加载预训练权重以避免网络问题
        # self.shared_encoder = timm.create_model(
        #     # model_name="resnet50", 
        #     model_name="pvt_v2_b4",
        #     pretrained=False, 
        #     in_chans=3, 
        #     features_only=True
        # )
        
        #self.shared_encoder = MyEncoderResNet(imagenet_pretrained=True) # 使用Res2Net50作为特征提取器，加载ImageNet预训练权重
        self.shared_encoder = MyEncoderPvt2() # 使用PVTv2-B4作为特征提取器，加载ImageNet预训练权重

        self.translayer = TransLayer(out_c=channel)
        
        # self.CAMV_layers = nn.ModuleList([
        #     CAMV(in_dim=channel, mm_size=size) 
        #     for size in [96, 48, 24, 12]
        # ])

        self.EFF_Layers = nn.ModuleList([
            EfficientFreqFusion(in_dim=channel, mm_size=size)
            for size in [96, 48, 24, 12]
        ])
        
        self.d5 = CFU(channel*2, num_groups=6, hidden_dim=channel)
        self.d4 = CFU(channel*2, num_groups=6, hidden_dim=channel)
        self.d3 = CFU(channel*2, num_groups=6, hidden_dim=channel)
        self.d2 = CFU(channel*2, num_groups=6, hidden_dim=channel)
        # self.d1 = CFU(channel*2, num_groups=6, hidden_dim=channel)

        
        self.NCD1 = NeighborConnectionDecoder(channel)

        # self.out_layer_00 = ConvBNReLU(channel, 32, 3, 1, 1)
        self.out_layer_00 = two_ConvBnRule(channel*2, 64)
        self.out_layer_01 = nn.Conv2d(64, 1, 1)

        
        self.classifier1 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.classifier2 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.classifier3 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.classifier4 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.classifier5 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)

        # self.rfb0_1 = RFB_modified(64, channel)
        self.myrfb0 = RFB_modified(64, channel)
        self.myrfb1 = RFB_modified(128, channel)
        self.myrfb2 = RFB_modified(320, channel)
        self.myrfb3 = RFB_modified(512, channel)

        self.mytrans0 = ConvBNReLU(channel*2, channel, 3, 1, 1)
        self.mytrans1 = ConvBNReLU(channel*2, channel, 3, 1, 1)
        self.mytrans2 = ConvBNReLU(channel*2, channel, 3, 1, 1)
        self.mytrans3 = ConvBNReLU(channel*2, channel, 3, 1, 1)

        self.MSA5 = MSA_module(dim = channel)
        self.MSA4 = MSA_module(dim = channel)
        self.MSA3 = MSA_module(dim = channel)
        self.MSA2 = MSA_module(dim = channel)

        # # ---- reverse stage ----
        # self.RS5 = ReverseStage(channel)
        # self.RS4 = ReverseStage(channel)
        # self.RS3 = ReverseStage(channel)
        # self.RS2 = ReverseStage(channel)

        # self.sa0 = SpatialAttention(3)
        
        self.dcn = DeformableAlign(channel)
        
        self.data_processor = DataProcessor()
    
    def encoder_translayer(self, x):
        feats = self.shared_encoder(x)
        return self.translayer(feats)
    
    def body(self, o, a1, a2):
        o_feats_ori = self.shared_encoder(o) #(batch_size, C, H, W) 0为浅层 3为深层

        # #print("输出类型：", o_feats_ori.keys())
        # print("####",o_feats_ori[0].shape)  # (1,64,192,192)
        # print("\n####",o_feats_ori[1].shape) # (1,256,96,96)
        # print("\n####",o_feats_ori[2].shape) # (1,512,48,48)
        # print("\n####",o_feats_ori[3].shape) # (1,1024,24,24)  
        # #print("\n####",o_feats_ori[4].shape) # (1,2048,12,12)
        # while 1:
        #     ...

        o_feats = self.translayer(o_feats_ori) # 统一通道数到64
        # 统一通道数到64，o_feats[3]为深层特征，o_feats[0]为浅层特征
        
        a1_ori = self.shared_encoder(a1)
        a1_feats_flip = [torch.flip(a1_temp,dims=[2]) for a1_temp in a1_ori] # 垂直翻转
        a1_feats = self.translayer(a1_feats_flip)
        a2_ori = self.shared_encoder(a2)
        a2_feats_flip = [torch.flip(a2_temp,dims=[3]) for a2_temp in a2_ori] # 水平翻转
        a2_feats = self.translayer(a2_feats_flip)

        # feat_diff = a1_feats_flip[0] - a2_feats_flip[0]
        # print("feat_diff形状为：", feat_diff.shape) #6
        # print("feat_diff的值：", feat_diff)
        # while 1:
        #     ...

        # a1_feats = self.encoder_translayer(a1)
        # a2_feats = self.encoder_translayer(a2) # channel = 64/32

        # DCT频域提取增强
        iterate = [0,1,2,3]
        dct_feats_list = []
        for i, o_f, a1_f, a2_f in zip(iterate, o_feats, a1_feats, a2_feats):
            dct_temp = self.EFF_Layers[i](o_f, a1_f, a2_f)
            dct_feats_list.append(dct_temp) # 5* (batchsize, 64, , ) 浅到深
        
        # aligned_a1_feats = []
        # aligned_a2_feats = []
        # for a1_f, a2_f in zip(a1_feats, a2_feats):
        #     a1_aligned = self.dcn(a1_f)
        #     a2_aligned = self.dcn(a2_f)
        #     aligned_a1_feats.append(a1_aligned)
        #     aligned_a2_feats.append(a2_aligned)
        
        # feats_list = []
        # for o_f, a1_f, a2_f in zip(o_feats, aligned_a1_feats, aligned_a2_feats):
        #     fused = (o_f + a1_f + a2_f) / 3.0
        #     feats_list.append(fused)

        # x1_rfb = self.myrfb0(o_feats_ori[1])  # 256 -> 64 96 96
        x1_rfb = self.myrfb0(o_feats_ori[0])  # 64 -> 64 96 96
        x2_rfb = self.myrfb1(o_feats_ori[1])  # 128 -> channel 48 48
        x3_rfb = self.myrfb2(o_feats_ori[2])  # 320 -> channel 24 24
        x4_rfb = self.myrfb3(o_feats_ori[3])  # 512 -> channel 12 12
        
        # 生成粗略的预测结果
        x0 = self.d5(torch.cat([x4_rfb, dct_feats_list[3]], dim=1)) # 深
        x0_fuse = self.mytrans0(x0) # 融合dct频域增强后的较深层特征
        x0_up = F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=False)  # (batchsize, channel, 24, 24)
        
        x1 = self.d4(x0_up + torch.cat([x3_rfb, dct_feats_list[2]], dim=1))
        x1_fuse = self.mytrans1(x1)
        x1_up = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False) # (batchsize, channel, 48, 48)
        
        x2 = self.d3(x1_up + torch.cat([x2_rfb, dct_feats_list[1]], dim=1))
        x2_fuse = self.mytrans2(x2)
        x2_up = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False) # (batchsize, channel, 96, 96)
        
        x3 = self.d2(x2_up + torch.cat([x1_rfb, dct_feats_list[0]], dim=1))
        x3_fuse = self.mytrans3(x3)
        # x3_up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False) # (batchsize, channel, 192, 192)
        P5 = F.interpolate(x3_fuse, scale_factor=0.125, mode='bilinear') # (batchsize, channel, 12, 12)


        # print("x4_rfb形状为：", x4_rfb.shape)
        # print("x3_rfb形状为：", x3_rfb.shape)
        # print("x2_rfb形状为：", x2_rfb.shape)
        # print("dct_feats_list[3]形状为：", dct_feats_list[3].shape)
        # print("dct_feats_list[2]形状为：", dct_feats_list[2].shape)
        # print("dct_feats_list[1]形状为：", dct_feats_list[1].shape)
        # print("x0_fuse形状为：", x0_fuse.shape)
        # print("x1_fuse形状为：", x1_fuse.shape)
        # print("x2_fuse形状为：", x2_fuse.shape)
        # while 1:
        #     ...

        # Neighbourhood Connected Decoder
        # S_g = self.NCD1(x4_rfb, x3_rfb, x2_rfb) # 深层特征->浅层特征 f4,f3,f2
        S_g = self.out_layer_01(self.out_layer_00(x3)) # 直接使用融合dct频域增强后的较浅层特征进行解码 (96,96)
        S_g = F.interpolate(S_g, scale_factor=0.5, mode='bilinear') # 使用融合dct频域增强后的较浅层特征进行解码
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')    # Sup-1 (bs, 1, 48, 48) -> (bs, 1, 352, 352)

        # ---- stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear') # (12,12)
        D4_feat = self.MSA5(P5, x4_rfb, guidance_g) # (12,12)
        D4_feat = F.interpolate(D4_feat, scale_factor=2, mode='bilinear') # (24,24)
        S_5 = self.classifier5(D4_feat)
        S_5_pred = F.interpolate(S_5, scale_factor=16, mode='bilinear')  # Sup-2 (bs, 1, 24, 24) -> (bs, 1, 352, 352)

        # ---- stage 4 ----
        # guidance_4 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        D3_feat = self.MSA4(D4_feat, x3_rfb, S_5) # (24,24)
        D3_feat = F.interpolate(D3_feat, scale_factor=2, mode='bilinear') # (24,24)
        S_4 = self.classifier4(D3_feat)
        S_4_pred = F.interpolate(S_4, scale_factor=8, mode='bilinear')  # Sup-3 (bs, 1, 48, 48) -> (bs, 1, 352, 352)

        # ---- stage 3 ----
        # guidance_3 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        D2_feat = self.MSA3(D3_feat, x2_rfb, S_4) # (48,48)
        D2_feat = F.interpolate(D2_feat, scale_factor=2, mode='bilinear') # (48,48)
        S_3 = self.classifier3(D2_feat)
        S_3_pred = F.interpolate(S_3, scale_factor=4, mode='bilinear')   # Sup-4 (bs, 1, 96, 96) -> (bs, 1, 352, 352)

        # ---- stage 2 ----
        D1_feat = self.MSA2(D2_feat, x1_rfb, S_3) # (96,96)
        D1_feat = F.interpolate(D1_feat, scale_factor=2, mode='bilinear') # (96,96)
        S_2 = self.classifier2(D1_feat)
        S_2_pred = F.interpolate(S_2, scale_factor=2, mode='bilinear')   # Sup-5 (bs, 1, 96, 96) -> (bs, 1, 352, 352)

        # print("S_g_pred形状为：", S_g_pred.shape)
        # print("S_5_pred的形状：", S_5_pred.shape) 
        # print("S_4_pred的形状：", S_4_pred.shape)
        # print("S_3_pred的形状：", S_3_pred.shape)
        # print("S_2_pred的形状：", S_2_pred.shape)
        # while 1:
        #     ...

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred, S_2_pred
    
    def train_forward(self, data):
        # data：(,3,384,384)
        if torch.is_tensor(data):
            processed = self.data_processor.process_batch_image(data)
        else:
            processed = data
        
        o = processed["image_o"].to(next(self.parameters()).device) if torch.is_tensor(processed["image_o"]) else processed["image_o"]
        a1 = processed["image_a1"].to(next(self.parameters()).device) if torch.is_tensor(processed["image_a1"]) else processed["image_a1"]
        a2 = processed["image_a2"].to(next(self.parameters()).device) if torch.is_tensor(processed["image_a2"]) else processed["image_a2"]
        
        if o.dim() == 3:
            o = o.unsqueeze(0)
        if a1.dim() == 3:
            a1 = a1.unsqueeze(0)
        if a2.dim() == 3:
            a2 = a2.unsqueeze(0)

        # print("o形状：", o.shape)
        # while 1:
        #     ...
        
        output = self.body(o, a1, a2) # 输入为384*384
        # return {"pred": output}
        return output
    
    def test_forward(self, data):
        
    
        if torch.is_tensor(data):
            processed = self.data_processor.process_batch_image(data)
        else:
            processed = data
        
        o = processed["image_o"].to(next(self.parameters()).device) if torch.is_tensor(processed["image_o"]) else processed["image_o"]
        a1 = processed["image_a1"].to(next(self.parameters()).device) if torch.is_tensor(processed["image_a1"]) else processed["image_a1"]
        a2 = processed["image_a2"].to(next(self.parameters()).device) if torch.is_tensor(processed["image_a2"]) else processed["image_a2"]
        
        if o.dim() == 3:
            o = o.unsqueeze(0)
        if a1.dim() == 3:
            a1 = a1.unsqueeze(0)
        if a2.dim() == 3:
            a2 = a2.unsqueeze(0)
        
        output = self.body(o, a1, a2)
        return output[-1].sigmoid() # 返回个预测图
    
    def cal_loss(self, outputs, targets):
        if isinstance(outputs, dict):
            pred = outputs.get('pred', list(outputs.values())[0])
        else:
            pred = outputs
        return pred