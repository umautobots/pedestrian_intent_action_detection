'''
Oct 7th
The original PIEPredict code extract VGG16 features and saved them to disk.
We read these features and add local bounding box to it.
'''
import torch, glob, os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import pdb

root = 'data/PIE_dataset/prepared_data/'
feature_root = 'data/PIE_dataset/saved_output/data/pie'

all_dirs = [x[0] for x in os.walk(os.path.join(root, 'image_patches'))]
print(all_dirs)
print(len(all_dirs))
pdb.set_trace()
for sub_dir in all_dirs:
    all_files = sorted(glob.glob(os.path.join(sub_dir,'*.pkl')))
    print("{}: {}".format(sub_dir, len(all_files)))
    vgg16_feature = {}
    for f in tqdm(all_files):
        split, sid, vid, file_name = f.split('/')[-4:]
        save_path = os.path.join(root, 'vgg16_features', '/'.join(f.split('/')[-4:-1]))
        save_file = os.path.join(save_path, f.split('/')[-1])
        feature_file = os.path.join(feature_root, split, 'features_context_pad_resize/vgg16_none', sid, vid, file_name)

        if not os.path.exists(feature_file):
            print(feature_file)
            continue
        if os.path.exists(save_file):
            continue
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # load local bounding box data:
        img_patch_data = pkl.load(open(f, 'rb'))
        vgg16_feature['local_bbox'] = np.array(img_patch_data['local_bbox']) 

        # load feature
        feature_data = pkl.load(open(feature_file, 'rb'))
        vgg16_feature['feature'] = np.array(feature_data) 
        pkl.dump(vgg16_feature, open(save_file, 'wb')) 