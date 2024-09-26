from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import traceback

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


from Recording_class import Recording
from Spheroid_class import Spheroid


class Batch:
    def __init__(self, file_list:list, output_path:Path, data_structure:str, params:dict) -> None:
        
        self.file_list = file_list
        self.output_path = Path(output_path)
        self.data_structure = data_structure
        self.params = params
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.path_error_log = self.output_path / 'error_spheroid.log'
        

    def __check_create_output_folder(self):
        raw_folder = 'Raw_Data' if self.data_structure == 'Raw_Data' else 'raw_microscopy_files'
        for f_path in self.file_list:
            folder_path = Path(str(f_path.parent).replace(raw_folder,'Analysis'))
            if not folder_path.is_dir():
                folder_path.mkdir(parents=True)
            
            folder_path = Path(str(f_path.parent).replace(raw_folder,'Analysis_Cache'))
            if not folder_path.is_dir():
                folder_path.mkdir(parents=True)


    def wrapper_fctn(self, f_path):
        print(f'Current file: {f_path}')
        rec = Recording(f_path, self.data_structure, self.params['seg_identifier'], ch_nuclei=self.params['ch_nuclei'], additional_seg_identifiers=self.params['additional_seg_identifiers'])
        if rec.path_seg is None:
            self.write_error(f_path, 'No segmentation found!')
            spher = {'Path': f_path, 'Error': 1}
            return spher
        
        spher = rec.load_spheroid()
        if spher is None:
            try:
                spheroid = Spheroid(rec, params=self.params)
                spher = spheroid.spher_to_dict()
                rec.save_spheroid(spher)

            except Exception as e: # most generic exception you can catch
                self.write_error(f_path, e, traceback.format_exc())
                spher = {'Path': f_path, 'Error': 1}
        else:
            print(f'Loaded from data: {f_path}')

        return spher


    def run(self, mode='single', n_threads=4):
        self.__check_create_output_folder()
        

        if mode == 'single':
            spher_list = [self.wrapper_fctn(file_path) for file_path in self.file_list]
        elif mode == 'parallel':
            with Pool(n_threads) as p:
                spher_list = p.map(self.wrapper_fctn, self.file_list)


        
        # Liste nach Fehlern absuchen und diese Fehlermeldungen irgendwie rausschreiben
        spher_list_errors = [x for x in spher_list if x['Error'] != 0]
        spher_list = [x for x in spher_list if x['Error'] == 0]

        errors_found = False if len(spher_list_errors)==0 else True
        result_text = 'Folder completed' if not errors_found else 'Folder completed with errors'
        print(f'\n{"_"*20}\n{result_text}\n{"_"*20}')
        for spher in spher_list_errors:
            print(f'Found error on: {spher["Path"]}')

        

        print('\nCalculate and save spheroid stats for folder')
        
        # Create stats for the cells of each spheroid
        for spher in spher_list:
            spher['Recording']['path_out_cells'].parent.mkdir(exist_ok=True)
            dsheet_cell_stats = pd.DataFrame.from_records([Batch.spher_cell_stats(blob) for blob in spher['Cells']])
            dsheet_cell_stats.to_csv(spher['Recording']['path_out_cells'], sep=';')
            dsheet_cell_stats.to_excel(str(spher['Recording']['path_out_cells']).replace('.csv','.xlsx'))

            if spher['Recording']['additional_seg_identifiers'] is not None:
                for props, name in zip(spher['Additional Props'], spher['Recording']['additional_seg_identifiers']):
                    out_path_add = Path(spher['Recording']['path_out_cells'])
                    out_path_add = out_path_add.parent / (out_path_add.stem + '_' + name + out_path_add.suffix)
                    dsheet_cell_stats = pd.DataFrame.from_records([Batch.spher_cell_stats(prop, basic=True) for prop in props])
                    dsheet_cell_stats.to_csv(out_path_add, sep=';')
                    dsheet_cell_stats.to_excel(str(out_path_add).replace('.csv','.xlsx'))


        # Create stats with each spheroid individually
        indv_dict_list = []
        if len(spher_list) != 0:
            for spher in spher_list:
                spher_stats = Batch.spher_group_stats([spher], spher['Path'])
                indv_dict_list.append(spher_stats)
                dsheet_spher = pd.DataFrame.from_records([spher_stats])
                dsheet_spher.to_csv(spher['Recording']['path_out_spher'], sep=';')
                dsheet_spher.to_excel(str(spher['Recording']['path_out_spher']).replace('.csv','.xlsx'))
            dsheet_spher_indv = pd.DataFrame.from_records(indv_dict_list)
        else:
            dsheet_spher_indv = pd.DataFrame()

        # Create stats with combined spheroids
        spher_group_list = [spher for spher in spher_list]
        if len(spher_group_list) != 0:
            spher_group_dict = Batch.spher_group_stats(spher_group_list, self.output_path)
            dsheet_spher_group = pd.DataFrame.from_records([spher_group_dict])
            dsheet_spher_group.to_csv(self.output_path / 'spher_group_stats.csv', sep=';')
            dsheet_spher_group.to_excel(self.output_path / 'spher_group_stats.xlsx')
        else:
            dsheet_spher_group = pd.DataFrame()
            
        self.spher_list = spher_list

        return dsheet_spher_indv, dsheet_spher_group, errors_found


    def write_error(self, f_path, message, traceback=None):
        print('Spheroid calculation error')
        print(message)
        with open(self.path_error_log, 'a') as err_file:
            if traceback is None:
                err_file.write(f'Spheroid calculation error {datetime.now()}\n' +
                            f'Error on file {f_path}:\n{message}\n\n')
            else:
                err_file.write(f'Spheroid calculation error {datetime.now()}\n' +
                            f'Error on file {f_path}:\n{message}\n{traceback}\n\n')

    def create_histograms(self, variable, var_name='', title='', density=True):
        '''
        Histograms
        # variable = '' # Dict names of the variable
        # var_name = '' # Dispayed names of the variable
        # title = '' # Displayed title
        # density = True # If the histogram should show the density
        '''
        # Get the min max range for the plot
        min_max_value = np.asarray([np.inf, 0])
        n_values = 0
        for spher in self.spher_list:
            n_values += len(spher['Cells'])
            qt_5 = np.quantile([variable(blob) for blob in spher['Cells']], 0.05)
            if qt_5 < min_max_value[0]: min_max_value[0] = qt_5
            qt_95 = np.quantile([variable(blob) for blob in spher['Cells']], 0.95)
            if qt_95 > min_max_value[1]: min_max_value[1] = qt_95

        # Create the Histogram for the specified variable
        values = np.zeros(n_values)
        spher_counter = 0
        b_i = 0
        for spher in self.spher_list:
            spher_counter += 1
            for blob in spher['Cells']:
                values[b_i] = variable(blob)
                b_i += 1

        if density:
            plt.hist(values, 50, min_max_value, density=True) # -> Sum=1
        else:
            plt.hist(values, 50, min_max_value)
        plt.xlabel(var_name)
        plt.ylabel('Probability') if density else plt.ylabel(f'Counts (in {spher_counter} images)')
        plt.title(title)
        plt.grid(True)
        plt.savefig(self.output_path / f'Histogram_{title}.png')
        plt.close()

    def create_2d_hist(self, variables, var_names, title=''):
        '''
        2D Histogram plots
        # variables = [] # List with the dict names of the variables
        # var_names = [] # List with the dispayed names of the variables
        # title = '' # Displayed title
        '''

        # Get the min max range for the plot
        if len(variables) == 2:
            min_max_value = np.asarray([[np.inf, 0], [np.inf, 0]])
        else:
            raise ValueError('Expected len of variables to be 2')

        n_values = 0
        for spher in self.spher_list:
            n_values += len(spher['Cells'])
            for v_i, var in enumerate(variables):
                qt_5 = np.quantile([var(blob) for blob in spher['Cells']], 0.05)
                if qt_5 < min_max_value[v_i,0]: min_max_value[v_i,0] = qt_5
                qt_95 = np.quantile([var(blob) for blob in spher['Cells']], 0.95)
                if qt_95 > min_max_value[v_i,1]: min_max_value[v_i,1] = qt_95

        # Create the scatter for the specified values
        values = np.zeros((n_values,len(variables)))
        b_i = 0
        for spher in self.spher_list:
            for blob in spher['Cells']:
                for v_i, var in enumerate(variables):
                    values[b_i, v_i] = var(blob)
                b_i += 1
        if len(variables)==2:
            plt.hist2d(values[:,0], values[:,1], (255, 255), cmap=plt.cm.jet)
            cbar = plt.colorbar()
        else:
            raise ValueError(f'variables can only contain 2 or 3 entrys, but contains {len(variables)}')
        plt.xlim(min_max_value[0,:])
        plt.ylim(min_max_value[1,:])
        plt.xlabel(var_names[0])
        plt.ylabel(var_names[1])
        plt.title('Title')
        plt.grid(True)
        plt.savefig(self.output_path / f'Scatter_{title}.png')
        plt.close()



    def create_scatterplots(self, variables, var_names, title=''):
        '''
        Scatter plots
        # variables = [] # List with the dict names of the variables
        # var_names = [] # List with the dispayed names of the variables
        # title = '' # Displayed title
        '''

        # Get the min max range for the plot
        if len(variables) == 2:
            min_max_value = np.asarray([[np.inf, 0], [np.inf, 0]])
        else:
            min_max_value = np.asarray([[np.inf, 0], [np.inf, 0], [np.inf, 0]])
        n_values = 0
        for spher in self.spher_list:
            n_values += len(spher['Cells'])
            for v_i, var in enumerate(variables):
                qt_5 = np.quantile([var(blob) for blob in spher['Cells']], 0.05)
                if qt_5 < min_max_value[v_i,0]: min_max_value[v_i,0] = qt_5
                qt_95 = np.quantile([var(blob) for blob in spher['Cells']], 0.95)
                if qt_95 > min_max_value[v_i,1]: min_max_value[v_i,1] = qt_95

        # Create the scatter for the specified values
        values = np.zeros((n_values,len(variables)))
        b_i = 0
        for spher in self.spher_list:
            for blob in spher['Cells']:
                for v_i, var in enumerate(variables):
                    values[b_i, v_i] = var(blob)
                b_i += 1
        if len(variables)==2:
            plt.scatter(values[:,0], values[:,1], s=0.2)
        elif len(variables)==3:
            sc = plt.scatter(values[:,0], values[:,1], c=values[:,2], vmin=min_max_value[2,0], vmax=min_max_value[2,1], s=0.2)
            cbar = plt.colorbar(sc)
            cbar.set_label(var_names[2], rotation=90)
        else:
            raise ValueError(f'variables can only contain 2 or 3 entrys, but contains {len(variables)}')
        plt.xlim(min_max_value[0,:])
        plt.ylim(min_max_value[1,:])
        plt.xlabel(var_names[0])
        plt.ylabel(var_names[1])
        plt.title(title)
        plt.grid(True)
        plt.savefig(self.output_path / f'Scatter_{title}.png')
        plt.close()


    @staticmethod
    def __new_dict_entry(out_dict, name, values, mode='mean', std=True, normalize=None):
        # out_dict = {}
        if mode == 'mean':
            if type(values) == list and len(values) == 0:
                out_dict[f'{name}'] = None
            elif type(values) == list and len(values) > 1:
                out_dict[f'{name} (mean)'] = np.mean(values)
                if std:
                    out_dict[f'{name} (std)'] = np.std(values)
            else:
                out_dict[f'{name}'] = values[0] if type(values)==list else values
        
        elif mode == 'count':
            if normalize is not None and normalize != 1:
                out_dict[f'{name} (mean)'] = values / normalize
            else:
                out_dict[f'{name}'] = values
        return out_dict

    @staticmethod
    def spher_group_stats(spher_list, info=None):
        cells_list = []
        additional_props_list = [[] for _ in spher_list[0]['Additional Props']]
        counter_spher = 0
        for spher in spher_list:
            cells_list += spher['Cells']
            for ind_add_props, additional_props in enumerate(spher['Additional Props']):
                additional_props_list[ind_add_props] += additional_props
            counter_spher += 1

        s_dict = {}
        
        s_dict['File_Path'] = info if info is not None else ''
        if len(spher_list) == 1: s_dict['Seg_Path'] = spher_list[0]['Recording']['path_seg']
        s_dict['Number of spheroids'] = counter_spher

        Batch.__new_dict_entry(s_dict, 'Volume [µm^3]', [spher['Volume_um'] for spher in spher_list])

        Batch.__new_dict_entry(s_dict, 'Diameter [µm]', [spher['Diameter_um'] for spher in spher_list])

        Batch.__new_dict_entry(s_dict, 'Density (no. nuclei/volume) [no. nuclei / µm^3]', [spher['Density_um'] for spher in spher_list])
        
        Batch.__new_dict_entry(s_dict, 'Volume void [µm^3]', [spher['Volume_void_um'] for spher in spher_list])
        Batch.__new_dict_entry(s_dict, 'Volume ratio void/total', [spher['Volume_ratio_void'] for spher in spher_list])

        for i in range(spher_list[0]['Number_of_channels']):          
            Batch.__new_dict_entry(s_dict, f'Signal mean ch{i+1}', [spher['Signal_mean'][i] for spher in spher_list])
            Batch.__new_dict_entry(s_dict, f'Signal mean fg ch{i+1}', [spher['Signal_mean_fg'][i] for spher in spher_list])

            Batch.__new_dict_entry(s_dict, f'Signal mean in ring inner ch{i+1}', [spher['Signal_mean_ring'][1,i] for spher in spher_list])
            Batch.__new_dict_entry(s_dict, f'Signal mean in ring middle ch{i+1}', [spher['Signal_mean_ring'][2,i] for spher in spher_list])
            Batch.__new_dict_entry(s_dict, f'Signal mean in ring outer ch{i+1}', [spher['Signal_mean_ring'][3,i] for spher in spher_list])
            #fg
            Batch.__new_dict_entry(s_dict, f'Signal mean in ring inner fg ch{i+1}', [spher['Signal_mean_ring_fg'][1,i] for spher in spher_list])
            Batch.__new_dict_entry(s_dict, f'Signal mean in ring middle fg ch{i+1}', [spher['Signal_mean_ring_fg'][2,i] for spher in spher_list])
            Batch.__new_dict_entry(s_dict, f'Signal mean in ring outer fg ch{i+1}', [spher['Signal_mean_ring_fg'][3,i] for spher in spher_list])


        Batch.__new_dict_entry(s_dict, 'Number of nuclei', len([1 for cell in cells_list]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'Number of nuclei - neg', len([1 for cell in cells_list if cell['cell_class'] == 0]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'Number of nuclei - pos', len([1 for cell in cells_list if cell['cell_class'] == 1]), mode='count', normalize=counter_spher)

        Batch.__new_dict_entry(s_dict, 'Volume of inner ring [µm^3]', [spher['Ring_volumes_um'][0] for spher in spher_list])
        Batch.__new_dict_entry(s_dict, 'Volume of middle ring [µm^3]', [spher['Ring_volumes_um'][1] for spher in spher_list])
        Batch.__new_dict_entry(s_dict, 'Volume of outer ring [µm^3]', [spher['Ring_volumes_um'][2] for spher in spher_list])
        Batch.__new_dict_entry(s_dict, 'Density of inner ring', [spher['Ring_densities_um'][0] for spher in spher_list])
        Batch.__new_dict_entry(s_dict, 'Density of middle ring', [spher['Ring_densities_um'][1] for spher in spher_list])
        Batch.__new_dict_entry(s_dict, 'Density of outer ring', [spher['Ring_densities_um'][2] for spher in spher_list])
        
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in inner ring', len([1 for cell in cells_list if cell['ring_nr'] == 1]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in middle ring', len([1 for cell in cells_list if cell['ring_nr'] == 2]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in outer ring', len([1 for cell in cells_list if cell['ring_nr'] == 3]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in inner ring - neg', len([1 for cell in cells_list if cell['ring_nr'] == 1 and cell['cell_class'] == 0]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in middle ring - neg', len([1 for cell in cells_list if cell['ring_nr'] == 2 and cell['cell_class'] == 0]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in outer ring - neg', len([1 for cell in cells_list if cell['ring_nr'] == 3 and cell['cell_class'] == 0]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in inner ring - pos', len([1 for cell in cells_list if cell['ring_nr'] == 1 and cell['cell_class'] == 1]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in middle ring - pos', len([1 for cell in cells_list if cell['ring_nr'] == 2 and cell['cell_class'] == 1]), mode='count', normalize=counter_spher)
        Batch.__new_dict_entry(s_dict, 'No. of nuclei in outer ring - pos', len([1 for cell in cells_list if cell['ring_nr'] == 3 and cell['cell_class'] == 1]), mode='count', normalize=counter_spher)

        Batch.__new_dict_entry(s_dict, 'Mean nuclei volume [µm] in inner ring', [cell['volume_um'] for cell in cells_list if cell['ring_nr'] == 1], std=False)
        Batch.__new_dict_entry(s_dict, 'Mean nuclei volume [µm] in middle ring', [cell['volume_um'] for cell in cells_list if cell['ring_nr'] == 2], std=False)
        Batch.__new_dict_entry(s_dict, 'Mean nuclei volume [µm] in outer ring', [cell['volume_um'] for cell in cells_list if cell['ring_nr'] == 3], std=False)
        
        Batch.__new_dict_entry(s_dict, 'Distance to hull [µm]', [cell['distance2hull_um'] for cell in cells_list])
        Batch.__new_dict_entry(s_dict, 'Distance to hull - neg [µm]', [cell['distance2hull_um'] for cell in cells_list if cell['cell_class'] == 0])
        Batch.__new_dict_entry(s_dict, 'Distance to hull - pos [µm]', [cell['distance2hull_um'] for cell in cells_list if cell['cell_class'] == 1])

        Batch.__new_dict_entry(s_dict, 'Distance to center [µm]', [cell['distance2center_um'] for cell in cells_list])
        Batch.__new_dict_entry(s_dict, 'Distance to center - neg [µm]', [cell['distance2center_um'] for cell in cells_list if cell['cell_class'] == 0])
        Batch.__new_dict_entry(s_dict, 'Distance to center - pos [µm]', [cell['distance2center_um'] for cell in cells_list if cell['cell_class'] == 1])

        if spher_list[0]['Recording']['additional_seg_identifiers'] is not None:
            for props, name in zip(additional_props_list, spher_list[0]['Recording']['additional_seg_identifiers']):
                Batch.__new_dict_entry(s_dict, f'Number of {name} objects', len([1 for _ in props]), mode='count', normalize=counter_spher)
                Batch.__new_dict_entry(s_dict, f'No. of {name} objects in inner ring', len([1 for prop in props if prop['ring_nr'] == 1]), mode='count', normalize=counter_spher)
                Batch.__new_dict_entry(s_dict, f'No. of {name} objects in middle ring', len([1 for prop in props if prop['ring_nr'] == 2]), mode='count', normalize=counter_spher)
                Batch.__new_dict_entry(s_dict, f'No. of {name} objects in outer ring', len([1 for prop in props if prop['ring_nr'] == 3]), mode='count', normalize=counter_spher)
                Batch.__new_dict_entry(s_dict, f'{name} objects Distance to hull [µm]', [prop['distance2hull_um'] for prop in props])
                Batch.__new_dict_entry(s_dict, f'{name} objects Distance to center [µm]', [prop['distance2center_um'] for prop in props])
        
        return s_dict

    @staticmethod
    def spher_cell_stats(cell, info=None, basic=False):
        # Ring number
        if cell['ring_nr'] == 1: ring = 'inner'
        elif cell['ring_nr'] == 2: ring = 'middle'
        elif cell['ring_nr'] == 3: ring = 'outer'
        elif cell['ring_nr'] == 0: ring = 'outside of spheroid'
        else: raise ValueError(f'Unknown ring number: {cell["ring_nr"]}. Must be one of [0,1,2,3].')

        c_dict = {}
        c_dict['Info'] = info if info is not None else ''
        
        c_dict['Distance to hull [µm]'] = cell['distance2hull_um']
        c_dict['Distance to center [µm]'] = cell['distance2center_um']
        c_dict['Ring location'] = ring
        
        if not basic:
            # Cell type
            if cell['cell_class'] == 1: cell_class = 'pos'
            elif cell['cell_class'] == 0: cell_class = 'neg'
            elif cell['cell_class'] == -1: cell_class = 'error'
            else: raise ValueError(f'Unknown cell type number: {cell["cell_class"]}. Must be one of [0,1].')

            c_dict['Cell type'] = cell_class
            c_dict['Volume [µm^3]'] = cell['volume_um']
            c_dict['Centroid (z,y,x) [px]'] = cell['centroid']
            c_dict['BoundingBox (zmin,y,x,zmax,y,x) [px]'] = cell['bbox']
            c_dict['Extent'] = cell['extent']
            c_dict['Filled volume [µm^3]'] = cell['filled_volume_um']
            c_dict['Label number'] = cell['label']
            c_dict['Major axis length [µm]'] = cell['major_axis_length']
            c_dict['Minor axis length [µm]'] = cell['minor_axis_length']
            c_dict['Moments'] = cell['moments']
            c_dict['Convex volume [µm]'] = cell['convex_volume_um']
            c_dict['Solidity'] = cell['solidity']

            n_ch = cell['features']['signal_mean'].size
            for key in cell['features'].keys():
                for i in range(n_ch):
                    c_dict[f'{key} ch{i+1}'] = cell['features'][key][i]
        
        return c_dict

    @staticmethod
    def combine_spher_results(out_path:Path, dsheet_spher_indv_list:list, dsheet_spher_group_list:list, folder_name_list:list):
        out_path.mkdir(exist_ok=True)
        
        # Create results with each spheroid individually
        if len(dsheet_spher_indv_list) != 0:
            if len(dsheet_spher_indv_list) != len(folder_name_list):
                raise ValueError(f'Lenght of both lists must be similar: {len(dsheet_spher_indv_list)}{len(folder_name_list)}')
            dsheet_spher_indv = pd.concat(dsheet_spher_indv_list)
            try:
                dsheet_spher_indv.to_csv(out_path / 'res_spher_individual_stats.csv', sep=';')
            except Exception as e:
                print(f'Error during creation of *.csv summary file:\n{e}')
            try:
                with pd.ExcelWriter(out_path / 'res_spher_individual_stats.xlsx') as writer:
                    for dsheet_spher_indv, folder_name in zip(dsheet_spher_indv_list, folder_name_list):
                        sheet_name = str(folder_name).replace('/',' ').replace('\\', ' ')
                        if sheet_name == '':
                            sheet_name = 'NN'
                        dsheet_spher_indv.to_excel(writer, sheet_name=sheet_name)
            except Exception as e:
                print(f'Error during creation of *.xlsx summary file:\n{e}')

        # Create stats with combined spheroids
        if len(dsheet_spher_group_list) != 0:
            dsheet_spher_group = pd.concat(dsheet_spher_group_list)
            try:
                dsheet_spher_group.to_csv(out_path / 'res_spher_group_stats.csv', sep=';')
            except Exception as e:
                print(f'Error during creation of combined *.csv summary file:\n{e}')
            try:
                dsheet_spher_group.to_excel(out_path / 'res_spher_group_stats.xlsx')
            except Exception as e:
                print(f'Error during creation of combined *.xlsx summary file:\n{e}')