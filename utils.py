from pathlib import Path
import pickle
import sys

def get_size(obj, seen=None):
    """
    Recursively finds size of python objects
    Origin: https://stackoverflow.com/a/38515297
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def check_for_pickle(f_path):
    f_path = Path(str(f_path).replace('Raw_Data', 'Analysis_Cache').replace('.tif', '.data'))
    if f_path.is_file():
        with open(f_path, 'rb') as input_file:
            return pickle.load(input_file)
    else:
        return None