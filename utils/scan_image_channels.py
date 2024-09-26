import os

import numpy as np
import tifffile as tiff


'''
Script to print image shapes in folders and subfolders.
'''

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


path = 'path/to/folder'
path_list, _ = get_folder_files_iter(path)
print_mean_shape = False
count=0
mean_shape = [0,0,0,0]
for f_path in path_list:
    shape = tiff.imread(f_path).shape
    print('Shape: {}, {}'.format(shape, f_path))
    if count == 0:
        mean_shape = shape
    else:
        mean_shape = np.add(mean_shape, shape)
    count +=1
if print_mean_shape:
    mean_shape = mean_shape/count
    print('mean_shape: {}'.format(mean_shape))