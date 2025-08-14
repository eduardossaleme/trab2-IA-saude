from .resnet import MyResnet
from .vggnet import MyVGGNet
from .mobilenet import MyMobilenet
import torchvision
from torchvision.models import ResNet50_Weights, VGG16_Weights, MobileNet_V2_Weights

_MODELS = ['resnet-50','vgg-16','mobilenet']


def set_model (model_name, num_class=6, neurons_reducer_block=0, freeze_conv=False, p_dropout=0.5):

    if model_name not in _MODELS:
        raise Exception("The model {} is not available!".format(model_name))

    model = None
    if model_name == 'resnet-50':
        model = MyResnet(torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT), num_class, freeze_conv=freeze_conv, 
                                                                p_dropout=p_dropout, neurons_reducer_block= neurons_reducer_block)

    elif model_name == 'vgg-16':
        model = MyVGGNet(torchvision.models.vgg16(weights= VGG16_Weights.DEFAULT), num_class, freeze_conv=freeze_conv, 
                                                                p_dropout=p_dropout, neurons_reducer_block= neurons_reducer_block)
    
    elif model_name == 'mobilenet':
        model = MyMobilenet(torchvision.models.mobilenet_v2(weights= MobileNet_V2_Weights.DEFAULT), num_class, freeze_conv=freeze_conv, 
                                                                p_dropout=p_dropout, neurons_reducer_block= neurons_reducer_block)


    return model


