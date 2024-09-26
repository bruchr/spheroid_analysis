import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter, sobel, binary_fill_holes
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.transform import rescale
import tifffile as tiff

import Model_class
from speroid_seg import spheroid_seg
from Recording_class import Recording


class Spheroid():
    def __init__(self, recording:Recording, params=None, verbose=False):
        self.img = recording.load_img() # Last dimension are channels [z,y,x,c]
        self.seg = recording.load_seg()
        if self.img is None:
            raise RuntimeError('No valid image')
        if self.seg is None:
            raise RuntimeError('No valid segmentation')
        
        self.px_size = recording.extract_px_size()
        self.seg_spher = recording.load_seg_spher()
        self.seg_spher_rings = recording.load_seg_spher_rings()
        # self.seg_spher_distance = recording.seg_spher_distance
        self.recording = recording
        self.params = params
        self.verbose = verbose

        # Functions to calculate
        self.img = self.__resize(self.img, segmentation=False)
        self.seg = self.__resize(self.seg, segmentation=True)

        self.additional_prop_list = []
        if recording.additional_seg_identifiers is not None:
            for add_seg_path in recording.additional_seg_paths:
                seg_tmp = self.__resize(recording.load_seg(add_seg_path), segmentation=True)
                self.additional_prop_list.append(regionprops(seg_tmp))
            del seg_tmp


        self.cells = regionprops(self.seg, self.img[:,:,:,recording.ch_nuclei])
        # Delete nuclei outside of range
        self.remove_outside_vol_range()
        self.n_nuclei = len(self.cells)

        self.recording.path_seg_spher.parent.mkdir(exist_ok=True)

        if self.seg_spher is None:
            if self.verbose: print('Calculating segmentation')
            self.seg_spher = spheroid_seg(self.seg)
            # tiff.imsave(self.recording.path_seg_spher, self.seg_spher)
        elif self.verbose:
            print('Loading segmentation from file')
        
        self.volume = np.count_nonzero(self.seg_spher)
        self.volume_um = self.volume * self.px_size[1]**3
        
         ### Ring Analysis
        if self.seg_spher_rings is None:
            if self.verbose: print('Calculating rings')
            self.__calculate_rings()
        elif self.verbose:
            print('Loading rings from file')
        
        self.__analyze_rings()
        self.__calculate_signal_sum_mean()
        self.seg_spher_rings = None
        
        self.__calculate_diameter()
        self.__analyze_distance2hull()
        self.__analyze_distance2center()
        self.seg_spher = None
        
        self.n_nuclei_in_core = 0
        for prop in self.cells:
            if prop.distance2hull >= 0: self.n_nuclei_in_core += 1
        self.density = np.divide(self.n_nuclei_in_core, self.volume)
        self.density_um = np.divide(self.n_nuclei_in_core, self.volume_um)
        self.volume_void =  self.volume - np.count_nonzero(self.seg)
        self.volume_void_um =  self.volume_void * self.px_size[1]**3
        self.v_ratio_void = self.volume_void/self.volume


        self.__calculate_cell_features()
        self.__classify_cells()

        # self.__compute_cell_zones()



    def check_param(self, param):
        return param in self.params and self.params[param] is not None and self.params[param] != 'None' and self.params[param] != 'none'

    def __resize(self, img, segmentation=False):
        scaling = self.px_size[0]/self.px_size[1]
        if segmentation:
            return rescale(img, (scaling,1,1), order=0, preserve_range=True).astype(np.uint16)
        else:
            return rescale(img, (scaling,1,1), order=1, channel_axis=-1, preserve_range=True).astype(np.uint8)

    def __calculate_diameter(self):
        n_slices = self.seg_spher.shape[0]
        d_list = np.zeros(n_slices)

        for ind_sl, sl in enumerate(range(n_slices)):
            vol = np.count_nonzero(binary_fill_holes(self.seg_spher[sl, ...]))
            d_list[ind_sl] = 2 * np.sqrt(vol/np.pi)

        self.spheroid_diameter = np.quantile(d_list, q=0.98)
        self.spheroid_diameter_um = self.spheroid_diameter * self.px_size[1]

    def __calculate_rings(self, n_rings=3):
        self.number_of_rings = n_rings
        self.seg_spher_rings = np.copy(self.seg_spher)
        ring = np.copy(self.seg_spher)
        ring_counter = n_rings-1
        ring_vol = [self.volume,]
        while ring_counter>0:
            ring = binary_erosion(ring)
            ring_tmp_vol = np.count_nonzero(ring)
            if ring_tmp_vol <= self.volume*(ring_counter/n_rings):
                self.seg_spher_rings[np.logical_and(ring==0, self.seg_spher_rings==1)] = ring_counter+1
                ring_vol.append(ring_tmp_vol)
                ring_counter -= 1
        for ind_r in range(len(ring_vol)-1):
            ring_vol[ind_r] -= ring_vol[ind_r + 1]
        ring_vol.reverse()
        self.ring_vol = ring_vol
        self.ring_vol_um = np.multiply(self.ring_vol, self.px_size[1]**3)
    
    def __calculate_signal_sum_mean(self):
        self.signal_sum = np.sum(self.img, axis=(0,1,2))
        self.signal_sum_per_nuclei = np.divide(self.signal_sum, self.n_nuclei)
        self.signal_sum_fg = np.zeros(self.img.shape[-1])
        self.signal_sum_fg_per_nuclei = np.zeros(self.img.shape[-1])
        self.signal_mean = np.mean(self.img, axis=(0,1,2))
        self.signal_mean_fg = np.zeros(self.img.shape[-1])
        # Ring 0 is background. Innermost is 1.
        self.signal_sum_ring = np.zeros((self.seg_spher_rings.max()+1, self.img.shape[-1]))
        self.signal_sum_ring_per_nuclei = np.zeros_like(self.signal_sum_ring)
        self.signal_sum_ring_fg = np.zeros_like(self.signal_sum_ring)
        self.signal_sum_ring_fg_per_nuclei = np.zeros_like(self.signal_sum_ring)
        self.signal_mean_ring = np.zeros((self.seg_spher_rings.max()+1, self.img.shape[-1]))
        self.signal_mean_ring_fg = np.zeros_like(self.signal_mean_ring)

        for c in range(self.img.shape[-1]):
            img_c = self.img[...,c]
            img_f = gaussian_filter(img_c, sigma=(0,2,2))#, preserve_range=True)
            img_seg = img_f >= self.params['channel_thresh'][c]
            
            self.signal_sum_fg[c] = np.sum(img_c[img_seg])
            self.signal_sum_fg_per_nuclei[c] = self.signal_sum_fg[c]/self.n_nuclei
            self.signal_mean_fg[c] = np.mean(img_c[img_seg])

            for ring_nr in range(self.signal_sum_ring.shape[0]):
                self.signal_sum_ring[ring_nr,c] = np.sum(img_c, where=(self.seg_spher_rings==ring_nr))
                self.signal_sum_ring_fg[ring_nr,c] = np.sum(img_c, where=np.logical_and(self.seg_spher_rings==ring_nr, img_seg))
                self.signal_mean_ring[ring_nr,c] = np.mean(img_c, where=(self.seg_spher_rings==ring_nr))
                self.signal_mean_ring_fg[ring_nr,c] = np.mean(img_c, where=np.logical_and(self.seg_spher_rings==ring_nr, img_seg))
                n_nuclei_ring = len([1 for cell in self.cells if cell['ring_nr'] == ring_nr])
                if n_nuclei_ring != 0:
                    self.signal_sum_ring_per_nuclei[ring_nr,c] = np.divide(self.signal_sum_ring[ring_nr,c], n_nuclei_ring)
                    self.signal_sum_ring_fg_per_nuclei[ring_nr,c] = np.divide(self.signal_sum_ring_fg[ring_nr,c], n_nuclei_ring)
                else:
                    self.signal_sum_ring_per_nuclei[ring_nr,c] = 0
                    self.signal_sum_ring_fg_per_nuclei[ring_nr,c] = 0

    def __analyze_distance2hull(self):
        edt = distance_transform_edt(self.seg_spher)
        for props in ([self.cells,] + self.additional_prop_list):
            centroids = np.round([prop.centroid for prop in props]).astype(np.uint16)
            distances = edt[tuple(centroids.T)]
            for ind, prop in enumerate(props):
                prop.distance2hull = distances[ind]

    def __analyze_distance2center(self):
        prop = regionprops(self.seg_spher)
        centroid = prop[0].centroid
        
        for props in ([self.cells,] + self.additional_prop_list):
            prop_centroids = np.asarray([prop.centroid for prop in props])
            distances = np.linalg.norm(np.subtract(prop_centroids,centroid), axis=1)
            for ind, prop in enumerate(props):
                prop.distance2center = distances[ind]

    def __classify_cells(self):
        if self.check_param('classification_threshold_variable') and self.check_param('classification_threshold_value'):
            thresh_channel = self.params['classification_threshold_variable_channel']
            for prop in self.cells:
                value = prop.features[self.params['classification_threshold_variable']][thresh_channel]
                if self.check_param('classification_threshold_normalization_variable'):
                    thresh_channel = self.params['classification_threshold_normalization_variable_channel']
                    value /= prop.features[self.params['classification_threshold_normalization_variable']][thresh_channel]
                prop.cell_class = value >= self.params['classification_threshold_value']
        
        elif 'classification_model' in self.params and self.params['classification_model'] is not None:
            try:
                model = Model_class.Model.load_model(self.params['classification_model'])
            except KeyError:
                print(f'Classification model could not be loaded. Path: {self.params["classification_model"]}')
                model = None

            if model is not None:
                X = model.extract_features(self, mode='inference')
                y_pred = model.inference(X)
                for ind, prop in enumerate(self.cells):
                    prop.cell_class = y_pred[ind]
            else:
                for prop in self.cells:
                    prop.cell_class = -1
        else:
            for prop in self.cells:
                prop.cell_class = -1


    def __calculate_cell_features(self):
        prototyping = False

        filtered_img = gaussian_filter(self.img.astype(np.float32),[1,1,1,0])
        # zone_without_nuclei = np.copy(self.seg_cell_zone)
        # zone_without_nuclei[self.seg!=0] = 0
        # edge_image = sobel(filtered_img)

        if prototyping:
            tiff.imsave('path/filtered_img.tif', filtered_img)
            # tiff.imsave('path/edge_img.tif', edge_image)

        if 'c_label_image' in self.params and self.params['c_label_image'] is not None:
            img_class_label = tiff.imread(self.params['c_label_image'])
        else:
            img_class_label = None

        iterations_dil = 4
        for prop in self.cells:
            prop.features = {}

            # Crop image for faster computation
            bb = np.asarray(prop.bbox)
            bb[0:3] = np.maximum(bb[0:3]-iterations_dil, 0)
            bb[3:6] = np.minimum(bb[3:6]+iterations_dil, self.seg.shape)
            seg_crop = self.seg[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
            img_crop = self.img[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5],:]
            img_crop_f = filtered_img[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5],:]
            img_crop_f_seg = img_crop_f >= 40
            label_crop = seg_crop == prop.label
            binary_dilation(label_crop, iterations=4, output=label_crop)
            if any(label_crop[seg_crop == 0]):
                # Normal case if nearby background values exist
                label_crop[seg_crop != 0] = False
            else:
                # In case that all nearby signals belong to cells
                label_crop[seg_crop == prop.label] = False

            values_nuc = self.img[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2], :]
            self.__extract_features(prop, 'signal', values_nuc)

            values_nuc = filtered_img[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2], :]
            self.__extract_features(prop, 'signal_filtered', values_nuc)
            
            coords_nearby = np.nonzero(label_crop)
            values_nearby = img_crop[coords_nearby]
            self.__extract_features(prop, 'nearby_signal', values_nearby)

            values_nearby = img_crop_f[coords_nearby]
            self.__extract_features(prop, 'nearby_signal_filtered', values_nearby)

            
            # values_nearby = img_crop_f_seg[coords_nearby]
            # prop.features['nearby_signal_class'] = np.any(values_nearby, axis=0)
            # prop.features['nearby_signal_seg_mean'] = np.mean(values_nearby, axis=0)

            # # Cell-Zone Signals
            # prop.features['zone_without_nuclei_signal_mean'] = np.mean(self.img[zone_without_nuclei == prop.label, :], axis=0)
            # prop.features['zone_without_nuclei_signal_std'] = np.std(self.img[zone_without_nuclei == prop.label, :], axis=0)
            # prop.features['zone_signal_mean'] = np.mean(self.img[self.seg_cell_zone == prop.label, :], axis=0)
            # prop.features['zone_signal_std'] = np.std(self.img[self.seg_cell_zone == prop.label, :], axis=0)
            
            # # Edge signal
            # prop.features['edge_signal_max'] = np.max(edge_image[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]])
            # prop.features['edge_signal_q95'] = np.quantile(edge_image[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]], 0.95)
            # prop.features['edge_signal_mean'] = np.mean(edge_image[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]])

            if img_class_label is not None:
                # Centroid based
                # prop.features['class_label'] = img_class_label[cent[0], cent[1], cent[2]]
                # Volume based
                prop.features['class_label'] = np.argmax(np.bincount(img_class_label[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]]))

    @staticmethod
    def __extract_features(prop, feature, values):
        prop.features[f'{feature}_mean'] = np.mean(values, axis=0)
        prop.features[f'{feature}_std'] = np.std(values, axis=0)
        prop.features[f'{feature}_median'] = np.median(values, axis=0)
        prop.features[f'{feature}_q59'] = np.quantile(values, 0.95, axis=0)
        prop.features[f'{feature}_max'] = np.max(values, axis=0)
        prop.features[f'{feature}_min'] = np.min(values, axis=0)
    

    def __compute_cell_zones(self):
        dist = distance_transform_edt(self.seg==0)
        mask = dist <= self.params['cell_zone_dist_max']
        self.seg_cell_zone = watershed(dist, self.seg, mask=mask)

    def __analyze_rings(self):
        if self.seg_spher_rings is None:
            self.__calculate_rings()
        for ind_props, props in enumerate(([self.cells,] + self.additional_prop_list)):
            nuclei_counter_rings = [0 for _ in range(self.number_of_rings)]
            for prop in props:
                # Centroid based
                # prop.ring_nr = seg_s_rings[int(prop.centroid[0]), int(prop.centroid[1]), int(prop.centroid[2])]
                # Volume based
                prop.ring_nr = np.argmax(np.bincount(self.seg_spher_rings[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]]))
                nuclei_counter_rings[prop.ring_nr - 1] += 1
            if ind_props == 0: # Only for nuclei
                self.ring_density = np.divide(nuclei_counter_rings, self.ring_vol)
                self.ring_density_um = np.divide(nuclei_counter_rings, self.ring_vol_um)

    def analyze_signal_sum(self):
        self.signal_sum = np.sum(self.img, axis=[0,1,2])

    def remove_outside_vol_range(self):
        # Volumes um to px
        max_vol = self.params['max_volume'] / (self.px_size[1]**3)  if self.check_param('max_volume') else np.inf
        min_vol = self.params['min_volume'] / (self.px_size[1]**3)  if self.check_param('min_volume') else 0
        
        len_cells_tmp = len(self.cells)
        # Delete nuclei outside of range
        cells_to_remove = [el for el in self.cells if el.area > max_vol or el.area < min_vol]
        for prop in cells_to_remove:
            self.seg[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]] = 0
        # Too big
        len_cells = len(self.cells)
        self.cells = [el for el in self.cells if el.area <= max_vol]
        self.n_nuclei_2big = len_cells - len(self.cells)
        # Too small
        len_cells = len(self.cells)
        self.cells = [el for el in self.cells if el.area >= min_vol]
        self.n_nuclei_2small = len_cells - len(self.cells)

        assert len_cells_tmp-len(cells_to_remove) == len(self.cells), 'Unequal amount of objects removed from list and segmentation'


    def spher_to_dict(self):
        return {
            'Px_size': self.px_size[1],
            'Volume_um': self.volume_um,
            'Volume_px': self.volume,
            'Diameter_um': self.spheroid_diameter_um,
            'Diameter_px': self.spheroid_diameter,
            'Density_um': self.density_um,
            'Density_px': self.density,

            'Volume_void_um': self.volume_void_um,
            'Volume_void_px': self.volume_void,
            'Volume_ratio_void': self.v_ratio_void,
            
            'Signal_sum': self.signal_sum,
            'Signal_sum_per_nuclei': self.signal_sum_per_nuclei,
            'Signal_sum_ring': self.signal_sum_ring,
            'Signal_sum_ring_per_nuclei': self.signal_sum_ring_per_nuclei,
            'Signal_sum_fg': self.signal_sum_fg,
            'Signal_sum_fg_per_nuclei': self.signal_sum_fg_per_nuclei,
            'Signal_sum_ring_fg': self.signal_sum_ring_fg,
            'Signal_sum_ring_fg_per_nuclei': self.signal_sum_ring_fg_per_nuclei,

            'Signal_mean': self.signal_mean,
            'Signal_mean_ring': self.signal_mean_ring,
            'Signal_mean_fg': self.signal_mean_fg,
            'Signal_mean_ring_fg': self.signal_mean_ring_fg,

            'Number_of_nuclei': self.n_nuclei,
            'Number_of_nuclei_in_core': self.n_nuclei_in_core,
            'Number_of_nuclei_del_too_big': self.n_nuclei_2big,
            'Number_of_nuclei_del_too_small': self.n_nuclei_2small,

            'Number_of_channels': self.img.shape[-1],
            'Number_of_rings': self.number_of_rings,

            'Ring_volumes_px': self.ring_vol,
            'Ring_volumes_um': self.ring_vol_um,
            'Ring_densities_px': self.ring_density,
            'Ring_densities_um': self.ring_density_um,

            'Cells': self.__cells2cells_light(self.cells),
            'Additional Props': [self.__cells2cells_light(props, basic=True) for props in self.additional_prop_list],
            
            'Path': self.recording.path_img,
            'Recording': self.recording.recording_to_dict(),
            
            'Error': 0
        }

    def __cells2cells_light(self, cells, basic=False):
        '''Used to achieve a smaller diskspace'''
        cells_light = []
        for blob in cells:
            # if any(np.subtract(blob.bbox[3:6], blob.bbox[0:3]) <= 1):
            #     print(f'2D cell element! Label number: {blob.label}')
            flat_cell = any(np.subtract(blob.bbox[3:6], blob.bbox[0:3]) <= 1)
            cell_light = {'distance2hull_um': blob.distance2hull * self.px_size[1],
                          'distance2center_um': blob.distance2center * self.px_size[1],
                          'ring_nr': blob.ring_nr,
            }
            if not basic:
                cell_light['cell_class'] = blob.cell_class
                cell_light['volume_um'] = blob.area * self.px_size[1]**3

                cell_light['bbox'] = blob.bbox
                cell_light['centroid'] = blob.centroid
                # cell_light['eccentricity'] = blob.eccentricity # not implemented for 3D
                cell_light['extent'] = blob.extent
                cell_light['filled_volume_um'] = blob.filled_area * self.px_size[1]**3
                cell_light['label'] = blob.label
                cell_light['major_axis_length'] = blob.major_axis_length * self.px_size[1]
                cell_light['minor_axis_length'] = blob.minor_axis_length * self.px_size[1] if not flat_cell else None
                cell_light['moments'] = blob.moments
                # cell_light['orientation'] = blob.orientation # not implemented for 3D
                # cell_light['perimeter'] = blob.perimeter # not implemented for 3D
                cell_light['slice'] = blob.slice
                cell_light['convex_volume_um'] = blob.convex_area * self.px_size[1]**3 if not flat_cell else np.nan
                # Seems to produce an error if 2D cell present or if area is to big (memory errror)
                cell_light['solidity'] = blob.solidity if not flat_cell else np.nan # Seems to produce an error if 2D cell present

                cell_light['mean_intensity'] = blob.mean_intensity
                cell_light['max_intensity'] = blob.max_intensity
                cell_light['min_intensity'] = blob.min_intensity

                cell_light['features'] = blob.features
            cells_light.append(cell_light)

        return cells_light
