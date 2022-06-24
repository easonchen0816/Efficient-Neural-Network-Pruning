import torch
import torchvision
from config import cfg
from models.resnet import resnet34

def get_resnet34_for_imagenet():
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=cfg.model.num_class, bias=True)
    return model
def get_resnet34_for_imagenet_resnet():
    model = resnet34(pretrained = True)
    return model


def get_model():
    pair = {
        'resnet34': get_resnet34_for_imagenet,
        'ir_resnet34': get_resnet34_for_imagenet_resnet
    }

    model = pair[cfg.model.name]()


    if cfg.base.cuda:
        model = model.cuda()

    if cfg.base.multi_gpus:
        model = torch.nn.DataParallel(model)
    return model
