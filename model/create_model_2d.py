from .deeplabv3_model import deeplabv3_resnet50
import torch
import torch
import torch.nn as nn

def create_deeplabv3_resnet50(aux=False, num_classes=11, pretrain_path=None,in_channel=3):
    model = deeplabv3_resnet50(aux=aux, num_classes=num_classes,in_channel=in_channel)

    if pretrain_path!=None:
        weights_dict = torch.load(pretrain_path, map_location='cpu')
        print("load weight from {} successed!".format(pretrain_path))
        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]
        if in_channel!=3:
            for k in list(weights_dict.keys()):
                if "backbone.conv1" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

def create_deeplabv3_resnet50_in3(aux=False, num_classes=11, pretrain_path=None,in_channel=3):
    model = deeplabv3_resnet50(aux=aux, num_classes=num_classes,in_channel=3)

    if pretrain_path!=None:
        weights_dict = torch.load(pretrain_path, map_location='cpu')
        print("load weight from {} successed!".format(pretrain_path))
        if num_classes != 21:

            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    if in_channel!=3:
        model=Xto3ChannelMODEL(inchannel=1,model=model)
    return model


class Xto3ChannelMODEL(nn.Module):
    def __init__(self,  inchannel=1,model=deeplabv3_resnet50):
        super(Xto3ChannelMODEL, self).__init__()

        self.channel_converter = nn.Conv2d(inchannel, 3, kernel_size=1, stride=1, padding=0)

        self.model = model

    def forward(self, x):
        # 将单通道输入转换为三通道
        x = self.channel_converter(x)
        # print(x.shape)
        x = self.model(x)['out']

        return x