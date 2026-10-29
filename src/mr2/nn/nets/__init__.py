"""Neural network architectures."""

from mr2.nn.nets.SNRAware import SNRAware
from mr2.nn.nets.BasicCNN import BasicCNN
from mr2.nn.nets.ConvDecoder import ConvDecoder
from mr2.nn.nets.DiT import DiT
from mr2.nn.nets.HourglassTransformer import HourglassTransformer
from mr2.nn.nets.MLP import MLP
from mr2.nn.nets.Restormer import Restormer
from mr2.nn.nets.SwinIR import SwinIR
from mr2.nn.nets.Uformer import Uformer
from mr2.nn.nets.UNet import AttentionGatedUNet, UNet
from mr2.nn.nets.VAE import VAE

__all__ = [
    'MLP',
    'VAE',
    'AttentionGatedUNet',
    'BasicCNN',
    'ConvDecoder',
    'DiT',
    'HourglassTransformer',
    'Restormer',
    'SNRAware',
    'SwinIR',
    'UNet',
    'Uformer',
]
