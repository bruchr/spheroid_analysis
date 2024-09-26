import itertools
import os
import pickle
import time as zime

from matplotlib import pyplot as cm
import numpy as np
from skimage import filters as sk_filters
from skimage.measure import regionprops
import tifffile as tiff
from tqdm import tqdm

from file_selection import folders_and_subfolders


def check_for_pickle(f_path):
    f_path = f_path.replace('Raw_Data', 'Analysis_Cache').replace('.tif', '.data')
    if os.path.isfile(f_path):
        with open(f_path, 'rb') as input_file:
            return pickle.load(input_file)
    else:
        return None


def get_img_and_seg(f_path, nuclei_channel):
    f_path_seg = f_path.replace('Raw_Data', 'Segmentation')
    f_path_seg = os.path.join(os.path.dirname(f_path_seg), 'pp_' + os.path.basename(f_path_seg))

    try:
        img = tiff.imread(f_path)[:, nuclei_channel, ...]
        seg = tiff.imread(f_path_seg)
    except Exception as e: # most generic exception you can catch
        img = None
        seg = None
    return img, seg

def normalize(vars, vmax=None, vmin=None):
    vars = np.asarray(vars)
    # if vmax is None: vmax = np.max(vars)
    # if vmin is None: vmin = np.min(vars)
    if vmax is None: vmax = np.quantile(vars,0.99)
    if vmin is None: vmin = np.quantile(vars,0.01)
    vars = (vars-vmin)*(1/vmax)
    return vars

def normalize_img(img, vmin=0, vmax=None):
    if vmin is None or vmax is None:
        img_f = sk_filters.gaussian(img, sigma=[0,1,1], preserve_range=True)
    if vmin is None:
        vmin = np.min(img_f)
    if vmax is None:
        vmax = np.max(img_f)
    img = (img-vmin)*(255/vmax)
    return img

def plot_in_img(img, seg, spher, feature, cm, vmax=None, vmin=None):
    vars = [feature(nuclei) for nuclei in spher['Cells']]
    # vars = normalize(vars, vmax, vmin)
    props = regionprops(seg)
    label_list = [prop.label for prop in props]

    img = img.astype(np.float32)
    overlay = np.zeros_like(img, dtype=np.float32)
    for ind, nuclei in enumerate(tqdm(spher['Cells'])):
        prop_ind = label_list.index(nuclei['label'])
        c = props[prop_ind].coords
        overlay[c[:,0], c[:,1], c[:,2],:] = np.multiply(cm(vars[ind])[:3],255)
    transperency = 0.5
    # img = img*(1-transperency) + overlay*transperency
    img = img + overlay*0.4
    return np.clip(img, 0, 255).astype(np.uint8)


def get_volume_group(nuclei):
    vol = nuclei['volume_um']
    if vol < 1000:
        g = 0
    elif vol >= 1000 and vol < 2000:
        g = 1
    else:
        g = 2
    return g

def cm_indiv(var):
    if var == 0:
        return (1,0,0,1)
    else:
        var = (var+1)/3
        return cm.viridis(var)

def sub_files(paths, pattern_list=[]):
    for pattern in pattern_list:
        paths =  [path for path in paths if pattern in path.split('Raw_Data')[1]]
    # a_string = "A string is more than its parts!"
    # matches = ["more", "wholesome", "milk"]
    # if any(x in a_string for x in matches):
    return paths
def calc_for_all():

    path_raw_data_list = [
        'path/to/input/folder1',
        'path/to/input/folder2',
    ]

    for path_raw_data in path_raw_data_list:
        feature = get_volume_group# lambda nuclei: (nuclei['volume_um'] > 1100) *255
        vmax, vmin = None, None
        colormap = cm_indiv #cm.bwr #cm.plasma #cm.autumn
        
        file_paths = folders_and_subfolders(path_raw_data)
        pattern = { 'ctypes': ['A549', 'KP4', 'Fibroblasten'],
                    'drugs': ['Doxorubicin', 'Paclitaxel'],
                    'times': ['96h', '144h'],
                    'mtypes': ['Monokultur','Kokultur'],
                    'doses': ['0uM','0.2uM','1uM','5uM'],
        }

        patterns = list(itertools.product(*pattern.values()))

        s = zime.time()
        for pat in patterns:
            if pat[0]=='Fibroblasten' and pat[3]=='Kokultur':
                continue
            paths = sub_files(file_paths, pat)
            for f_path in paths:
                ch_name = 'Ki67' if '29.08.2019' in f_path else 'Coll'
                save_path = f_path.replace('Raw_Data','Overlays').replace('.tif','__overlay_size_{}.tif'.format(ch_name))
                channel = 0 if '29.08.2019' in f_path else 1
                img, seg = get_img_and_seg(f_path, channel)
                if img is not None and seg is not None:
                    img = normalize_img(img)
                    img = np.stack((img,)*3, axis=-1)
                    spher = check_for_pickle(f_path)

                    img = plot_in_img(img, seg, spher, feature, colormap, vmax, vmin)

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    tiff.imsave(save_path, img)

if __name__ == "__main__":
    # f_path = 'path/to/image.tif'
    # save_path = 'path/to/output/image.tif'
    # nuclei_channel = 1
    # feature = get_volume_group # lambda nuclei: (nuclei['volume_um'] > 1100) *255
    # vmax, vmin = None, None
    # colormap = cm_indiv #cm.bwr #cm.plasma #cm.autumn


    # img, seg = get_img_and_seg(f_path, nuclei_channel)
    # img = normalize_img(img)
    # img = np.stack((img,)*3, axis=-1)
    # spher = check_for_pickle(f_path)

    # img = plot_in_img(img, seg, spher, feature, colormap, vmax, vmin)

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # tiff.imsave(save_path, img)

    calc_for_all()