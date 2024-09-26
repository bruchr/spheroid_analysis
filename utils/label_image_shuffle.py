import os

import numpy as np
from skimage.measure import label, regionprops
import tifffile as tiff


path_in_list = [
    'path/to/image.tif',
]

for path_in in path_in_list:
    path_out = path_in.replace('.tif','_shuffled.tif')

    seg = tiff.imread(path_in)
    seg = label(seg)
    props = regionprops(seg)

    ind_list_new = np.arange(1, len(props)+1)
    np.random.shuffle(ind_list_new)

    seg_ = np.zeros_like(seg, dtype=np.uint16)

    for ind, prop in enumerate(props):
        c = prop.coords
        seg_[c[:,0], c[:,1], c[:,2]] = ind_list_new[ind]

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    tiff.imsave(path_out, seg_)
