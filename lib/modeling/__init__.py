from .conv3d_based.act_intent import ActionIntentionDetection as Conv3dModel
from .rnn_based.model import ActionIntentionDetection as RNNModel
from .relation.relation_embedding import RelationEmbeddingNet

def make_model(cfg):
    if cfg.MODEL.TYPE == 'conv3d':
        model = Conv3dModel(cfg)
    elif cfg.MODEL.TYPE == 'rnn':
        model = RNNModel(cfg)
    elif cfg.MODEL.TYPE == 'relation':
        model = RelationEmbeddingNet(cfg)
    else:
        raise NameError("model type:{} is unknown".format(cfg.MODEL.TYPE)) 

    return model
