import os
from pathlib import Path
import sys
import re

import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog , QDialog, QMessageBox, QLineEdit, QWidget
import tifffile as tiff

from analysis import run as run_analysis
from file_selection import folders_and_subfolders

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    end_result = pyqtSignal(str)
    def __init__(self, params):
        super().__init__()
        self.params = params

    def start(self):

        run_analysis(self.params, results_callback=self.end_result.emit, progress_callback=self.progress.emit)
        
        self.progress.emit(100)
        self.finished.emit()


class Window(QDialog):

    def __init__(self):
        super(Window, self).__init__()
        uic.loadUi('gui_design.ui', self)
        
        self.input_folder = None
        self.model_path = None
        self.threads = 0
        
        self.min_vol = -1
        self.max_vol = -1

        self.nuclei_channel = 0
        self.channel_thresholds = ''

        self.n_channels = None

        self.add_seg_ident_fields = []

        self.setup_fields()

    def setup_fields(self)->None:
        self.label_input_folder.setText(self.input_folder)
        self.label_model_path.setText(self.model_path)
        self.btn_unselect_model.hide()
        
        self.spin_threads.setValue(self.threads)
        self.label_status.setText('')

        self.edit_min_vol.setText(str(self.min_vol))
        self.edit_max_vol.setText(str(self.max_vol))

        self.spin_nuc_ch.setValue(self.nuclei_channel)
        self.edit_ch_thresh.setText(str(self.channel_thresholds))

        self.btn_input_folder.clicked.connect(self.load_folder)
        self.btn_model_path.clicked.connect(self.load_model_file)
        self.btn_unselect_model.clicked.connect(self.remove_model_file)
        self.btn_start_analysis.clicked.connect(self.start_analysis)
        self.btn_add_seg_ident_plus.clicked.connect(self.add_seg_ident_add)
        self.btn_add_seg_ident_minus.clicked.connect(self.add_seg_ident_rem)

        self.cbox_data_structure.addItems(['Raw_Data', 'Cell_ACDC'])
        self.cbox_data_structure.setCurrentIndex(1)

        self.progressBar.setDisabled(True)
        self.setFocus()

        self.widget_list = self.findChildren(QWidget)

        for widget in self.widget_list:
            if not widget.objectName() in ['label_data_structure', 'cbox_data_structure', 'btn_input_folder']:
                widget.setDisabled(True)

    def fill_classification_cboxes(self):
        feature_list = ['None']
        for feature in ['signal', 'signal_filtered', 'nearby_signal', 'nearby_signal_filtered']:
            for method in ['mean', 'std', 'median', 'q95', 'max', 'min']:
                feature_list.append(f'{feature}_{method}')
        ch_list = [str(ch) for ch in range(self.n_channels)]

        self.cbox_cell_classification_f1.addItems(feature_list)
        self.cbox_cell_classification_f2.addItems(feature_list)
        self.cbox_cell_classification_f1_ch.addItems(ch_list)
        self.cbox_cell_classification_f2_ch.addItems(ch_list)

    def load_folder(self):
        f_path = QFileDialog.getExistingDirectory(self, 'Select input folder', '../')
        self.input_folder = f_path if os.path.exists(f_path) else None

        if self.input_folder is not None:
            if self.cbox_data_structure.currentText() == 'Raw_Data':
                pattern = 'Raw_Data'
            elif self.cbox_data_structure.currentText() == 'Cell_ACDC':
                pattern = 'raw_microscopy_files'
            test_files = folders_and_subfolders(self.input_folder, pattern=pattern)
            if len(test_files)>0:
                file_path = test_files[0]
                img_shape = tiff.imread(file_path).shape
                self.n_channels = img_shape[1] if len(img_shape)==4 else 1
                self.spin_nuc_ch.setMaximum(self.n_channels-1)
                self.edit_ch_thresh.setText(", ".join(str(0)*self.n_channels))
                self.fill_classification_cboxes()

            for widget in self.widget_list:
                if not widget.objectName() in ['progressBar']:
                    widget.setDisabled(False)
                
            self.label_input_folder.setText(self.input_folder)
            self.label_input_folder.repaint()
    
    def load_model_file(self):
        f_path, _ = QFileDialog.getOpenFileName(self, 'Select classification model', '../', "Tiff files (*.tiff *.tif)")
        if os.path.exists(f_path):
            self.model_path = f_path
            self.btn_unselect_model.show()

            self.label_model_path.setText(Path(self.model_path).name)
            self.label_model_path.repaint()

    def remove_model_file(self):
        self.model_path = None
        self.btn_unselect_model.hide()
        self.label_model_path.setText(self.model_path)
        self.label_model_path.repaint()

    def add_seg_ident_add(self):
        self.add_seg_ident_fields.append(QLineEdit())
        self.vert_layout_add_seg_ident.addWidget(self.add_seg_ident_fields[-1])
    
    def add_seg_ident_rem(self):
        if len(self.add_seg_ident_fields) >= 1:
            tmp = self.add_seg_ident_fields.pop()
            self.vert_layout_add_seg_ident.removeWidget(tmp)
            tmp.deleteLater()
            tmp = None
    
    def get_params(self)->dict:
        params = {}
        params['data_structure'] = self.cbox_data_structure.currentText()
        params['input_folder'] = self.input_folder
        params['out_path_res'] = self.input_folder
        
        
        params['params'] = {}
        params['params']['ch_nuclei'] = self.spin_nuc_ch.value()
        try:
            min_vol = int(self.edit_min_vol.text())
        except:
            min_vol = -1
            self.edit_min_vol.setText(str(min_vol))
        try:
            max_vol = int(self.edit_max_vol.text())
        except:
            max_vol = -1
            self.edit_max_vol.setText(str(max_vol))
        params['params']['min_volume'] = min_vol if min_vol!=-1 else 0
        params['params']['max_volume'] = max_vol if max_vol!=-1 else np.inf

        channel_thresh = re.split('; |, |;|,|\*|\n', self.edit_ch_thresh.text())
        try:
            params['params']['channel_thresh'] = [int(c_thresh) for c_thresh in channel_thresh]
        except:
            self.edit_ch_thresh.setText(", ".join(str(0)*self.n_channels))
            params['params']['channel_thresh'] = [0 for _ in range(self.n_channels)]

        params['params']['seg_identifier'] = self.edit_seg_identifier.text()

        params['params']['classification_model'] = self.model_path
        params['params']['classification_threshold_variable'] = self.cbox_cell_classification_f1.currentText()
        params['params']['classification_threshold_variable'] = self.cbox_cell_classification_f2.currentText()
        params['params']['classification_threshold_variable_channel'] = self.cbox_cell_classification_f1_ch.currentIndex()
        params['params']['classification_threshold_normalization_variable_channel'] = self.cbox_cell_classification_f2_ch.currentIndex()
        cell_classification_threshold = self.edit_cell_classification_threshold.text()
        if cell_classification_threshold == '':
            params['params']['classification_threshold_value'] = None
        else:
            try:
                params['params']['classification_threshold_value'] = float(cell_classification_threshold)
            except:
                params['params']['classification_threshold_value'] = None
                self.edit_cell_classification_threshold.setText('')
        
        params['max_threads'] = self.spin_threads.value()
        params['parallel_mode'] = params['max_threads'] > 1

        params['params']['additional_seg_identifiers'] = None if len(self.add_seg_ident_fields) == 0 else []
        for add_seg_ident in self.add_seg_ident_fields:
            params['params']['additional_seg_identifiers'].append(add_seg_ident.text())
        
        return params

    def start_analysis(self):
        if self.input_folder is None:
            print('Please select an input folder!')
            msgBox = QMessageBox()
            msgBox.setText("Please select an input folder!")
            msgBox.exec()
            return
        params = self.get_params()

        self.btn_start_analysis.setEnabled(False)
        self.progressBar.setValue(0)
        self.progressBar.setDisabled(False)
        self.progressBar.repaint()

        self.thread = QThread()
        self.worker = Worker(params)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.start)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: self.btn_start_analysis.setEnabled(True))
        # self.thread.finished.connect(lambda: self.label_status.setText('Run completed!'))
        self.worker.progress.connect(lambda val: self.progressBar.setValue(val))
        self.worker.end_result.connect(lambda val: self.label_status.setText(val))

        self.label_status.setText('Running...')
        self.thread.start()



if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())