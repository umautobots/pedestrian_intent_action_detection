import torch
import torch.nn.functional as F
from torch import nn

from .roi_align import ROIAlign
import pdb
class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio, canonical_level=4):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        # lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        # lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        # self.map_levels = LevelMapper(lvl_min, lvl_max, canonical_level=canonical_level)

    def convert_to_roi_format(self, boxes):
        if isinstance(boxes, list):
            concat_boxes = torch.cat([b.bbox for b in boxes], dim=0)
        else:
            concat_boxes = torch.cat([b for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois
    
    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)

        if num_levels == 1:
            return self.poolers[0](x, rois)

        levels = self.map_levels(boxes)        

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
                        (num_rois, num_channels, output_size, output_size),
                        dtype=dtype,
                        device=device,
        )
        no_grad_level = []
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            if len(idx_in_level) <= 0:
                no_grad_level.append(level)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)
        return result, no_grad_level


def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    return pooler
