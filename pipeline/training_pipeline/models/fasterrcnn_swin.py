"""
A lot of scripts borrowed/adapted from Detectron2.
https://github.com/facebookresearch/detectron2/blob/38af375052d3ae7331141bc1a22cfa2713b02987/detectron2/modeling/backbone/backbone.py#L11

https://github.com/oloooooo/faster_rcnn_swin_transformer_detection/tree/master

"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torchvision

from functools import partial
from torchvision.models.detection import FasterRCNN

import sys
sys.path.append('/hdd/yuchen/pipeline/training_pipeline')

from models.layers import (
    Backbone, 
    PatchEmbed, 
    Block, 
    get_abs_pos,
    get_norm,
    Conv2d,
    LastLevelMaxPool
)
from models.utils import _assert_strides_are_log2_contiguous


from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

sys.path.append('/hdd/yuchen/pipeline/training_pipeline/models')

from faster_rcnn_swin_transformer_detection import transforms
from faster_rcnn_swin_transformer_detection.network_files import BackboneWithFPN
from faster_rcnn_swin_transformer_detection.swin_transformer import swin_t, Swin_T_Weights, swin_b, Swin_B_Weights


def create_model(num_classes=81, pretrained=True, coco_model=False, model_setting = 'base_from_mae'):
    # print(net)
    
    # model_layers = net.blocks
    # new_layers = nn.ModuleList([layer for i, layer in enumerate(model_layers) if i not in range(20,24)])
    # net.blocks = new_layers

    print('current model setting:', model_setting)

    if model_setting == 'default':
        backbone = swin_t(weights=Swin_T_Weights.DEFAULT).features
        in_channels_list = [96, 96 * 2, 96 * 2 * 2, 96 * 2 * 2 * 2]
        
    elif model_setting == 'base':
        backbone = swin_b(weights=Swin_B_Weights.DEFAULT).features
        in_channels_list = [128, 128 * 2, 128 * 2 * 2, 128 * 2 * 2 * 2]
        
    elif model_setting == 'tiny_from_mae':
        backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        ckpt = torch.load('/hdd/yuchen/satdata/weights/swinmae150.pth')
        backbone.load_state_dict(ckpt, strict=False)
        backbone = backbone.features
        in_channels_list = [96, 96 * 2, 96 * 2 * 2, 96 * 2 * 2 * 2]

    elif model_setting == 'base_from_mae':
        backbone = swin_b(weights=Swin_B_Weights.DEFAULT)
        ckpt = torch.load('/hdd/yuchen/satdata/weights/swinmae100_base_005lr_aug.pth')
        backbone.load_state_dict(ckpt, strict=False)
        backbone = backbone.features
        in_channels_list = [128, 128 * 2, 128 * 2 * 2, 128 * 2 * 2 * 2]

    elif model_setting == 'tiny_from_dist':
        backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        student_model_path = "/hdd/yuchen/satdata/weights/temp/student_model_weights_5.pth"
        ckpt = torch.load(student_model_path)
        backbone = backbone.features
        backbone = model_weight_conversion(backbone, ckpt)
        in_channels_list = [96, 96 * 2, 96 * 2 * 2, 96 * 2 * 2 * 2]

    elif model_setting == 'base_from_dist':
        backbone = swin_b(weights=Swin_B_Weights.DEFAULT)
        student_model_path = "/hdd/yuchen/satdata/weights/temp/student_model_weights_1.pth"
        ckpt = torch.load(student_model_path)
        backbone = backbone.features
        backbone = model_weight_conversion(backbone, ckpt)
        in_channels_list = [128, 128 * 2, 128 * 2 * 2, 128 * 2 * 2 * 2]

        
    elif model_setting == 'large':
        print('not implemented')
        pass
        
    else: 
        print(model_setting, ' does not exist')

    return_nodes = {'1': '0', '2': '1', '4': '2', '6': '3'}

    backbone = BackboneWithFPN(backbone,
                               return_layers=return_nodes,
                               # in_channels_list=[96, 96 * 2, 96 * 2 * 2, 96 * 2 * 2 * 2],
                               in_channels_list=in_channels_list,
                               
                               out_channels=256,
                               extra_blocks=LastLevelMaxPool())
    
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    model = FasterRCNN(backbone,
                       num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler, box_head=box_head, rpn_head=rpn_head
                       )

    return model

def model_weight_conversion(backbone, ckpt):
    backbone[0][0].weight = torch.nn.Parameter(ckpt['swin.embeddings.patch_embeddings.projection.weight'])
    backbone[0][0].bias = torch.nn.Parameter(ckpt['swin.embeddings.patch_embeddings.projection.bias'])
    backbone[0][2].weight = torch.nn.Parameter(ckpt['swin.embeddings.norm.weight'])
    backbone[0][2].bias = torch.nn.Parameter(ckpt['swin.embeddings.norm.bias'])

    for idx in [(1,0),(3,1),(5,2),(7,3)]:
        b = idx[0]
        swin = idx[1]
        backbone[b][0].norm1.weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.layernorm_before.weight'])
        backbone[b][0].norm1.bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.layernorm_before.bias'])
        
        backbone[b][0].attn.qkv.weight =  torch.nn.Parameter(torch.stack([ckpt[f'swin.encoder.layers.{swin}.blocks.0.attention.self.query.weight'],
                                         ckpt[f'swin.encoder.layers.{swin}.blocks.0.attention.self.key.weight'],
                                        ckpt[f'swin.encoder.layers.{swin}.blocks.0.attention.self.value.weight']]).flatten(0,1))
        
        backbone[b][0].attn.qkv.bias =  torch.nn.Parameter(torch.stack([ckpt[f'swin.encoder.layers.{swin}.blocks.0.attention.self.query.bias'],
                                         ckpt[f'swin.encoder.layers.{swin}.blocks.0.attention.self.key.bias'],
                                        ckpt[f'swin.encoder.layers.{swin}.blocks.0.attention.self.value.bias']]).flatten(0,1))
            
        backbone[b][0].attn.proj.weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.attention.output.dense.weight'])
        backbone[b][0].attn.proj.bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.attention.output.dense.bias'])
        
        backbone[b][0].norm2.weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.layernorm_after.weight'])
        backbone[b][0].norm2.bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.layernorm_after.bias'])
        
        backbone[b][0].mlp[0].weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.intermediate.dense.weight'])
        backbone[b][0].mlp[0].bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.intermediate.dense.bias'])
        backbone[b][0].mlp[3].weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.output.dense.weight'])
        backbone[b][0].mlp[3].bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.0.output.dense.bias'])
        
        backbone[b][1].norm1.weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.layernorm_before.weight'])
        backbone[b][1].norm1.bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.layernorm_before.bias'])
    
        backbone[b][1].attn.qkv.weight =  torch.nn.Parameter(torch.stack([ckpt[f'swin.encoder.layers.{swin}.blocks.1.attention.self.query.weight'],
                                         ckpt[f'swin.encoder.layers.{swin}.blocks.1.attention.self.key.weight'],
                                        ckpt[f'swin.encoder.layers.{swin}.blocks.1.attention.self.value.weight']]).flatten(0,1))
        
        backbone[b][1].attn.qkv.bias =  torch.nn.Parameter(torch.stack([ckpt[f'swin.encoder.layers.{swin}.blocks.1.attention.self.query.bias'],
                                         ckpt[f'swin.encoder.layers.{swin}.blocks.1.attention.self.key.bias'],
                                        ckpt[f'swin.encoder.layers.{swin}.blocks.1.attention.self.value.bias']]).flatten(0,1))
        
        backbone[b][1].attn.proj.weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.attention.output.dense.weight'])
        backbone[b][1].attn.proj.bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.attention.output.dense.bias'])
    
        backbone[b][1].norm2.weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.layernorm_after.weight'])
        backbone[b][1].norm2.bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.layernorm_after.bias'])
        
        backbone[b][1].mlp[0].weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.intermediate.dense.weight'])
        backbone[b][1].mlp[0].bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.intermediate.dense.bias'])
        backbone[b][1].mlp[3].weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.output.dense.weight'])
        backbone[b][1].mlp[3].bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.blocks.1.output.dense.bias'])
        
        if b != 7:
            backbone[b+1].reduction.weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.downsample.reduction.weight'])
            backbone[b+1].norm.weight = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.downsample.norm.weight'])
            backbone[b+1].norm.bias = torch.nn.Parameter(ckpt[f'swin.encoder.layers.{swin}.downsample.norm.bias'])

    return backbone

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(81, pretrained=True)
    summary(model)