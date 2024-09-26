from pathlib import Path


def folders_and_subfolders(path:Path, pattern:list=None):
    path = Path(path)
    file_paths = []
    if type(pattern) != list and pattern is not None:
        pattern = [pattern]
    for file_path in path.glob('**/*.tif'):
        if pattern is None:
            file_paths.append(file_path)
        elif all([pat in str(file_path) for pat in pattern]):
            file_paths.append(file_path)

    return file_paths


def seperate_folders(file_paths:list):
    # Erstelle Liste mit allen Ordnern
    folder_dict = {}
    for file_path in file_paths:
        file_path = Path(file_path)
        dir_name = file_path.parent
        # Add folder to the list
        if dir_name not in folder_dict:
            folder_dict[dir_name] = [file_path]
        else:
            folder_dict[dir_name].append(file_path)

    return folder_dict


if __name__ == "__main__":
    path = Path('path/to/folder')
    file_paths_list = folders_and_subfolders(path, pattern='Raw_Data')
    folder_dict = seperate_folders(file_paths_list)
    print(folder_dict.keys())