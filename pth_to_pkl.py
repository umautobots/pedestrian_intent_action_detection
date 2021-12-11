'''
Oct 6th
We first saved 3*224*224 patches as .pth files which was too big.
Run this script to convert the .pth files to .pkl files to save space and time.
'''
import torch, glob, os
from tqdm import tqdm
import numpy as np
import pickle as pkl

root = 'data/PIE_dataset/prepared_data/'

all_dirs = [x[0] for x in os.walk(root)]
print(all_dirs)
print(len(all_dirs))
for sub_dir in tqdm(all_dirs):
    all_files = glob.glob(os.path.join(sub_dir,'*.pth'))
    print("{}: {}".format(sub_dir, len(all_files)))
    for f in all_files:
        save_file = f[:-4]+'.pkl'
        if os.path.exists(save_file):
            continue
        data = torch.load(f) 
        data['img_patch'] = np.array(data['img_patch']) 
        data['local_bbox'] = np.array(data['local_bbox']) 
        pkl.dump(data, open(save_file, 'wb')) 
        os.remove(f)