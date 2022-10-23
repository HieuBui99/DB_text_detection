import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_parameters
from modules.repvgg import RepVGG
from modules.segmentation_head import DBHead
from modules.segmentation_body import FPN

segmentation_body_dict = {'FPN': FPN}
segmentation_head_dict = {'DBHead': DBHead}


class MiniEncoderRepVGG(nn.Module):
    def __init__(self, deploy=False):
        super().__init__()
        # num_blocks = [2, 4, 14, 1]
        # width_multiplier = [0.75, 0.75, 0.75, 2.5]

        # num_blocks = [2, 4, 8, 1]
        # width_multiplier = [1, 1, 1, 1]
        num_blocks = [2, 4, 14, 1]
        self.width_multiplier = [0.75, 0.75, 0.75, 2.0]
        backbone = RepVGG(num_blocks=num_blocks,
                          num_classes=2,
                          width_multiplier=self.width_multiplier,
                          override_groups_map=None,
                          deploy=deploy)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = \
            backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4

    def forward(self, x):

        x1 = self.layer0(x)
        # print(x1.shape)
        x2 = self.layer1(x1)
        # print(x2.shape)
        x3 = self.layer2(x2)
        # print(x3.shape)
        x4 = self.layer3(x3)
        # print(x4.shape)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


class RepVGGDBTextModel(nn.Module):
    def __init__(self, pretrained=False, deploy=False):
        super().__init__()
        backbone_name = "repvgg"
        segmentation_body_name = "FPN"
        segmentation_head_name = "DBHead"
        inner_channels = 256

        self.backbone = MiniEncoderRepVGG(deploy=deploy)
        # backbone_out = [16, 32, 64, 128]
        # backbone_out = [32, 64, 128, 256]
        # backbone_out = [64, 128, 256, 512]
        backbone_out = [int(48 * self.backbone.width_multiplier[0]),
                        int(96 * self.backbone.width_multiplier[1]),
                        int(192 * self.backbone.width_multiplier[2]),
                        int(384 * self.backbone.width_multiplier[3])]

        self.segmentation_body = segmentation_body_dict[
            segmentation_body_name](backbone_out,
                                    inner_channels=inner_channels)
        self.segmentation_head = segmentation_head_dict[
            segmentation_head_name](self.segmentation_body.out_channels,
                                    out_channels=2)

        self.name = '{}_{}_{}'.format(backbone_name, segmentation_body_name,
                                      segmentation_head_name)

    def forward(self, x):
        """
        :return: TRAIN mode: prob_map, threshold_map, appro_binary_map
        :return: EVAL mode: prob_map, threshold_map
        """
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_body_out = self.segmentation_body(backbone_out)
        segmentation_head_out = self.segmentation_head(segmentation_body_out)
        y = F.interpolate(segmentation_head_out,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True)

        return y


if __name__ == "__main__":
    out_model = RepVGGDBTextModel(deploy=False)
    print(count_parameters(out_model))
