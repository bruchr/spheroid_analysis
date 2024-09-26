from pathlib import Path
import pickle

import numpy as np
import tifffile as tiff


class Recording:
    def __init__(self, img_path:Path, data_structure:str, seg_identifier:str=None, ch_nuclei:int=0, additional_seg_identifiers:list=None):
        self.path_img = Path(img_path)
        self.ch_nuclei = ch_nuclei
        self.seg_identifier = seg_identifier
        self.additional_seg_identifiers = additional_seg_identifiers

        if data_structure == 'Raw_Data':
            path_seg, base_name = self.find_segmentations(data_structure, seg_identifier)

            self.path_seg = path_seg
            self.path_seg_spher = Path(str(self.path_img.parent).replace('Raw_Data','Analysis_Cache')) / ('pp_' + base_name.replace('.tif','_spher.tif'))
            self.path_seg_spher_rings = Path(str(self.path_seg_spher).replace('.tif','_rings.tif'))
            # self.path_seg_spher_distance = self.path_seg_spher.replace('.tif','_distance.tif')

            self.path_data = Path(str(self.path_img.parent).replace('Raw_Data','Analysis_Cache')) / base_name.replace('.tif','.data')
            self.path_out_cells = Path(str(self.path_img.parent).replace('Raw_Data','Analysis')) / base_name.replace('.tif', '_data_cells.csv')
            self.path_out_spher = Path(str(self.path_img.parent).replace('Raw_Data','Analysis')) / base_name.replace('.tif', '_data_spher.csv')
            # self.path_result_folder = str(self.path_img.parent).replace('Raw_Data','Analysis')

        elif data_structure == 'Cell_ACDC':
            path_seg, base_name_corr = self.find_segmentations(data_structure, seg_identifier)

            if path_seg is not None:
                self.path_seg = path_seg
                
                self.path_seg_spher = path_seg.parents[1] / 'Cache' / (path_seg.stem + '_spher.tif')
                self.path_seg_spher_rings = path_seg.parents[1] / 'Cache' / (path_seg.stem + '_rings.tif')

                self.path_data = path_seg.parents[1] / 'Cache' / (base_name_corr + '.data')
                self.path_out_cells = path_seg.parents[1] / 'Analysis' / (base_name_corr + '_data_cells.csv')
                self.path_out_spher = path_seg.parents[1] / 'Analysis' / (base_name_corr + '_data_spher.csv')
                
            else: # Not required for data_structure Raw_Data as the other paths are not based on the segmentation path
                self.path_seg = None
                self.path_seg_spher = None
                self.path_seg_spher_rings = None
                self.path_data = None
                self.path_out_cells = None
                self.path_out_spher = None
        else:
            raise ValueError(f'Parameter "data_structure" must be either "Raw_Data" or "Cell_ACDC", but was {data_structure}')
        
        if additional_seg_identifiers is not None:
            self.additional_seg_paths = []
            for add_seg_ident in additional_seg_identifiers:
                path_seg, _ = self.find_segmentations(data_structure, add_seg_ident)
                self.additional_seg_paths.append(path_seg)
        else:
            self.additional_seg_paths = None
    
    def find_segmentations(self, data_structure:str, identifier:str):
        if data_structure == 'Raw_Data':
            path = self.path_img.parent
            base_name = self.path_img.name
            if identifier is None: identifier = ''
            path_seg = Path(str(path).replace('Raw_Data','Segmentation')) / (identifier + base_name)
            if not path_seg.is_file():
                path_seg = None
            return path_seg, base_name
        
        elif data_structure == 'Cell_ACDC':        
            path_group = self.path_img.parents[1]
            base_name = self.path_img.stem
            base_name_corr = base_name.replace('.', '_')
            if identifier is not None:
                # if identifier != '': identifier += '*' # Wildcard after identifier
                seg_paths = [p for p in path_group.rglob(base_name_corr + '_s*_segm*' + identifier + '.npz')]
                if len(seg_paths) == 0: # Try with .tif
                    seg_paths = [p for p in path_group.rglob(base_name_corr + '_s*_segm*' + identifier + '.tif')]

                if len(seg_paths)==1:
                    path_seg = seg_paths[0]
                elif len(seg_paths) == 0:
                    path_seg = None
                else:
                    ind_seg_path = np.argmax([len(str(seg_path)) for seg_path in seg_paths])
                    path_seg = seg_paths[ind_seg_path]
            else:
                path_seg = None
            return path_seg, base_name_corr

    def print_all_path(self):
        print(f'path_img: {self.path_img}')
        print(f'path_seg: {self.path_seg}')
        print(f'path_seg_spher: {self.path_seg_spher}')
        print(f'path_seg_spher_rings: {self.path_seg_spher_rings}')
        # print(f'path_seg_spher_distance: {self.path_seg_spher_distance}')

    def extract_px_size(self):
        with tiff.TiffFile(self.path_img) as tif:
            x_res = tif.pages[0].tags['XResolution'].value
            x_res = x_res[1]/x_res[0]
            spacing = float(tif.pages[0].tags['ImageDescription'].value.split('spacing=')[1].split('\n')[0])
            sp_unit = tif.pages[0].tags['ImageDescription'].value.split('unit=')[1].split('\n')[0]
            if sp_unit != 'micron':
                raise RuntimeError(f'Unit of px_size not supported: {sp_unit}')
            return [spacing, x_res, x_res]

    def load_img(self):
        if self.path_img.is_file():
            img = tiff.imread(self.path_img)
            if img.ndim == 3:
                img = img[..., None]
            else:
                img = np.moveaxis(img, 1,-1)
        else:
            img = None
        return img

    def load_seg(self, seg_path:Path=None):
        if seg_path is None:
            seg_path = self.path_seg
        if seg_path.is_file():
            if seg_path.suffix == '.tif':
                return tiff.imread(seg_path)
            elif seg_path.suffix == '.npz':
                data = np.load(seg_path)
                if len(data.files) > 1:
                    raise RuntimeError(f'Too many segmentation outouts in .npz file: {seg_path}')
                return data[data.files[0]]
        else:
            return None

    def load_seg_spher(self):
        return tiff.imread(self.path_seg_spher) if self.path_seg_spher.is_file() else None

    def load_seg_spher_rings(self):
        return tiff.imread(self.path_seg_spher_rings) if self.path_seg_spher_rings.is_file() else None

    # def load_seg_spher_distance(self):
    #     return tiff.imread(self.path_seg_spher_distance) if self.path_seg_spher_distance.is_file() else None

    def load_spheroid(self):
        if self.path_data.is_file():
            with open(self.path_data, 'rb') as input_file:
                return pickle.load(input_file)
        else:
            return None
    
    def save_spheroid(self, spher):
        with open(self.path_data, 'wb') as output_file:
            pickle.dump(spher, output_file, pickle.HIGHEST_PROTOCOL)


    def recording_to_dict(self):
        rec = {
            'path_img': self.path_img,
            'path_seg': self.path_seg,
            
            'path_seg_spher': self.path_seg_spher,
            'path_seg_spher_rings': self.path_seg_spher_rings,

            'path_data': self.path_data,
            'path_out_cells': self.path_out_cells,
            'path_out_spher': self.path_out_spher,
            'ch_nuclei': self.ch_nuclei,

            'seg_identifier': self.seg_identifier,
            'additional_seg_paths': self.additional_seg_paths,
            'additional_seg_identifiers': self.additional_seg_identifiers,

        }
        return rec