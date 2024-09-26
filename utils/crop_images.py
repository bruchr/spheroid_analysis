import os
import time

import numpy as np
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import regionprops, label
import tifffile as tiff


'''
Script to crop 3D images based on the content of the nuclei channel.
The metadata is copied to the cropped images so that the px size is not lost.
'''

def imread(path):
    with tiff.TiffFile(path) as tif:
        img = tif.asarray()
        mdata_ij = tif.imagej_metadata
        x_res = tif.pages[0].tags['XResolution'].value
        x_res = x_res[1]/x_res[0]
        z_res = tif.pages[0].tags['ImageDescription'].value.split('spacing=')[1].split('\n')[0]
        mdata = {
            'mdata_ij': mdata_ij,
            'x_res': x_res,
            'z_res': z_res,
        }
    return img, mdata

def imwrite(path, data, mdata):
    res = (1/mdata['x_res'], 1/mdata['x_res'])
    tiff.imwrite(path, data, imagej=True, resolution=res, metadata=mdata['mdata_ij'])

def get_folder_files_iter(path_to_data):
    file_paths = []
    file_paths_out = []
    base_folder = os.path.basename(path_to_data)
    for dirpath, _, files in os.walk(path_to_data):
            for f in files:
                if f.endswith('.tif'):
                    file_path = os.path.join(dirpath, f)
                    path_out = file_path.replace(base_folder, base_folder+os.path.sep + 'Raw_Data')
                    file_paths.append(file_path)
                    file_paths_out.append(path_out)
    return file_paths, file_paths_out

def crop_img_comp(img_r, ch, padding=(10,50,50)):
    img = img_r[:,ch,:,:]
    s = time.time()
    img = gaussian(img, sigma=(0,2,2))
    d = time.time()-s; print('Duration Gaussian: {:.3f} s'.format(d))
    
    s = time.time()
    img_thresh = threshold_otsu(img)
    d = time.time()-s; print('Duration Thresh: {:.3f} s'.format(d))

    s = time.time()
    props = regionprops(label(img > img_thresh))
    d = time.time()-s; print('Duration Thresholding+Regionprops: {:.3f} s'.format(d))
    
    # # Only biggest prop
    # area = [prop.area for prop in props]
    # prop = props[np.argmax(area)]
    # bb = list(prop.bbox)

    bb = np.asarray([10000,10000,10000, 0,0,0])
    for prop in props:
        if prop.area > 250:
            bb[0:3] = np.minimum(bb[0:3], prop.bbox[0:3])
            bb[3:6] = np.maximum(bb[3:6], prop.bbox[3:6])


    padding = (10, 50, 50)
    for i in range(3):
        bb[i] = np.maximum(0, bb[i]-padding[i])
        bb[i+3] = np.minimum(img.shape[i], bb[i+3]+padding[i])


    img_r = img_r[bb[0]: bb[3], :, bb[1]: bb[4], bb[2]: bb[5]]

    return img_r



if __name__=='__main__':

    in_path = 'path/to/folder'
    file_path_data, file_path_data_out = get_folder_files_iter(in_path)
    for f_in, f_out in zip(file_path_data, file_path_data_out):
        print(f_in)
        print(f_out)
        os.makedirs(os.path.dirname(f_out), exist_ok=True)
        img_r, mdata = imread(f_in)

        img_r = crop_img_comp(img_r, ch=1)

        imwrite(f_out, img_r, mdata)

