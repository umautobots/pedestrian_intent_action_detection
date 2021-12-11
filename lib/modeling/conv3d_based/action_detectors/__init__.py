from .i3d import InceptionI3d
from .c3d import C3D
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18

_MODEL_NAMES_ = {
'I3D': InceptionI3d,
'C3D': C3D,
'R3D_18': r3d_18,
'MC3_18': mc3_18,
'R2+1D_18': r2plus1d_18,
}

def make_model(model_name, num_classes, pretrained=True):
    if model_name in _MODEL_NAMES_:
        return _MODEL_NAMES_[model_name](num_classes=num_classes, pretrained=pretrained)
    else:
        valid_model_names = list(_MODEL_NAMES_.keys())
        raise ValueError('The model name is required to be one of {}, but got {}.'.format(valid_model_names, model_name))
    

