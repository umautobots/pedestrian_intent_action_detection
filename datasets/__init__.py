from .PIE import PIEDataset
from .JAAD import JAADDataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from .build_samplers import make_data_sampler, make_batch_data_sampler
import collections
import pdb

__DATASET_NAME__ = {
    'PIE': PIEDataset,
    'JAAD': JAADDataset
}
def make_dataloader(cfg, split='train', distributed=False, logger=None):
    is_train = split == 'train'
    if split == 'test':
        batch_size = cfg.TEST.BATCH_SIZE
    else:
        batch_size = cfg.SOLVER.BATCH_SIZE
    dataloader_params ={
            "batch_size": batch_size,
            "shuffle":is_train,
            "num_workers": cfg.DATALOADER.NUM_WORKERS,
            "collate_fn": collate_dict,
            }
    
    dataset = make_dataset(cfg, split)
    if is_train and cfg.DATALOADER.ITERATION_BASED:
        sampler = make_data_sampler(dataset, shuffle=is_train, distributed=distributed, is_train=is_train, weighted=cfg.DATALOADER.WEIGHTED!='none')
        batch_sampler = make_batch_data_sampler(dataset, 
                                                sampler, 
                                                aspect_grouping=False, 
                                                batch_per_gpu=batch_size,
                                                max_iters=cfg.SOLVER.MAX_ITERS, 
                                                start_iter=0, 
                                                dataset_name=cfg.DATASET.NAME)
        dataloader =  DataLoader(dataset, 
                                num_workers=cfg.DATALOADER.NUM_WORKERS, 
                                batch_sampler=batch_sampler,collate_fn=collate_dict)
    else:
        dataloader = DataLoader(dataset, **dataloader_params)
    if hasattr(logger, 'info'):
        logger.info("{} dataloader: {}".format(split, len(dataloader)))
    else:
        print("{} dataloader: {}".format(split, len(dataloader)))
    return dataloader


def make_dataset(cfg, split):
    return __DATASET_NAME__[cfg.DATASET.NAME](cfg, split)

def collate_dict(batch):
    '''
    batch: a list of dict
    '''
    if len(batch) == 0:
        return batch
    elem = batch[0]
    collate_batch = {}
    all_keys = list(elem.keys())
    for key in all_keys:
        # e.g., key == 'bbox' or 'neighbors_st' or so
        if elem[key] is None:
            collate_batch[key] = None
        # elif isinstance(elem, collections.abc.Sequence):
        #     if len(elem) == 4: # We assume those are the maps, map points, headings and patch_size
        #         scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
        #         if heading_angle[0] is None:
        #             heading_angle = None
        #         else:
        #             heading_angle = torch.Tensor(heading_angle)
        #         map = scene_map[0].get_cropped_maps_from_scene_map_batch(scene_map,
        #                                                                  scene_pts=torch.Tensor(scene_pts),
        #                                                                  patch_size=patch_size[0],
        #                                                                  rotation=heading_angle)
        #         return map
        #     transposed = zip(*batch)
        #     return [collate(samples) for samples in transposed]
        elif isinstance(elem[key], collections.abc.Mapping):
            # We have to dill the neighbors structures. Otherwise each tensor is put into
            # shared memory separately -> slow, file pointer overhead
            # we only do this in multiprocessing
            neighbor_dict = {sub_key: [b[key][sub_key] for b in batch] for sub_key in elem[key]}
            collate_batch[key] = dill.dumps(neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
        elif isinstance(elem[key], list):
            # NOTE: Nov 16, traffic objetcs number is not constant thus we use list to distinguish from tensor.
            if key == 'image_files':
                collate_batch[key] = [b[key] for b in batch]
            else:
                collate_batch[key] = [b[key][0] for b in batch]
        else:
            collate_batch[key] = default_collate([b[key] for b in batch])
    return collate_batch