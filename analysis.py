import argparse
import json
from pathlib import Path
import time

from Batch_class import Batch
from file_selection import folders_and_subfolders, seperate_folders

def check_parameters(params):
    """
    Checks if all necessary params are given and returns None.
    If params are missing, the key/name of the first one is returned.

    Args:
        params: A dict containing all parameters for the simulation process

    Returns:
        None if no key is missing, otherwise the key/name (str) of the first missing parameter
    """
    necessary_keys = [
        "data_structure",
        "input_folder", "out_path_res",
        "params",
        "parallel_mode"
    ]
    necessary_keys_params = [
        "ch_nuclei",
        "min_volume",
        "max_volume",
        "channel_thresh",
        "seg_identifier"
    ]

    for key in necessary_keys:
        if key not in params:
            raise KeyError(f'Key {key} not found in settings file!')
    for key in necessary_keys_params:
        if key not in params["params"]:
            raise KeyError(f'Key params[{key}] not found in settings file!')
    
    for key in ['input_folder', 'out_path_res']:
        params[key] = Path(params[key])

    if not 'max_threads' in params:
        params['max_threads'] = 4

    if not 'additional_seg_identifiers' in params:
        params['additional_seg_identifiers'] = None

    return params


def run(params:dict, results_callback:callable=None, progress_callback:callable=None) -> None:
    params = check_parameters(params)
    
    if params['data_structure'] == 'Raw_Data':
        raw_folder = 'Raw_Data'
    elif params['data_structure'] == 'Cell_ACDC':
        raw_folder = 'raw_microscopy_files'
    else:
        raise ValueError(f'Parameter "data_structure" must be either "Raw_Data" or "Cell_ACDC", but was {params["data_structure"]}')

    # Combine data in each folder
    file_paths = folders_and_subfolders(params['input_folder'], pattern=raw_folder)
    n_imgs = len(file_paths)
    folder_dict = seperate_folders(file_paths)
    path_batch_list = [folder for folder in folder_dict]
    
    # # Quickest run through all files. However, folderstructure is not used.
    # path_batch_list = [input_folder,]


    dsheet_spher_indv_list, dsheet_spher_group_list, folder_name_list = [], [], []
    errors_found_global = False
    for ind, path_batch in enumerate(path_batch_list):
        
        s = time.time()
        
        out_path = Path(str(path_batch).replace(raw_folder, 'Analysis'))
        file_paths = folders_and_subfolders(path_batch, pattern=raw_folder)
        batch = Batch(file_paths, out_path, params['data_structure'], params['params'])

        if params['parallel_mode']:
            dsheet_spher_indv, dsheet_spher_group, errors_found = batch.run(mode='parallel', n_threads=params['max_threads'])
        else:
            dsheet_spher_indv, dsheet_spher_group, errors_found = batch.run(mode='single')
        
        dsheet_spher_indv_list.append(dsheet_spher_indv)
        dsheet_spher_group_list.append(dsheet_spher_group)
        folder_name_list.append(path_batch.relative_to(params['input_folder']).parent)

        if errors_found:
            errors_found_global = True

        d = time.time()-s; print(f'Analysis took {d:.0f} s -> {d/60:.2f} min -> {d/3600:.2f} h\n\n')

        if progress_callback is not None:
            progress_callback(int(round(len(file_paths)/n_imgs)))

    if params['out_path_res'] is not None:
        Batch.combine_spher_results(params['out_path_res'], dsheet_spher_indv_list, dsheet_spher_group_list, folder_name_list)

    results_text = 'Run completed!' if not errors_found_global else 'Run completed with errors!'
    print(results_text)
    if results_callback is not None:
        results_callback(results_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Analysis Pipeline',
                    description='Extracts various features from provided 3D cell images and their corresponding segmentations.')
    parser.add_argument('settings', default='settings.json', help='Path to the settings file. If not provided "./settings.json" will be used.', nargs='?')
    args = parser.parse_args()

    settings_file = Path(args.settings)
    print(f'Settings file: {settings_file}')

    if not settings_file.is_file():
        raise ValueError(f'Settings file does not exist: {settings_file}')
    with open(settings_file, 'r') as json_settings:
        params = json.load(json_settings)

    run(params)