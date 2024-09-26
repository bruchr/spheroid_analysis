import pickle
import time

import numpy as np
from sklearn import svm, metrics


class Model():
    def __init__(self):
        self.clf = svm.SVC(kernel='linear')
        self.is_trained = False

    def train(self, X, y):
        self.clf.fit(X, y)
        self.is_trained = True

    def test(self, X, y):
        y_pred = self.clf.predict(X)
        print('Model Accuracy: {:.3f} %'.format(metrics.accuracy_score(y, y_pred)*100))

    def inference(self, X):
        if not self.is_trained:
            raise Exception('Model is not trained. Use train() to train the model first.')
        y_pred = self.clf.predict(X)
        return y_pred

    
    def extract_features(spher, feat_list: list, mode='inference'):

        class_label = lambda blob: blob['features']['class_label']

        if mode=='inference':
            samples = spher['Cells']
        else:
            samples = [blob for blob in spher['Cells'] if class_label(blob) != 0]
        
        n_obs = len(samples)
        n_feat = len(feat_list)
        
        X = np.zeros((n_obs, n_feat))
        y = None if mode=='inference' else np.zeros(n_obs)

        for ind_o, blob in enumerate(samples):
            for ind_f, feat in enumerate(feat_list):
                X[ind_o, ind_f] = feat(blob)
                if y is not None:
                    y[ind_o] = class_label(blob) -1

        if y is None:
            return X
        else:
            return X, y

    @staticmethod
    def extract_features_batch(spher_list: list, feat_list: list, mode='inference'):
        X_list = []
        y_list = []
        for spher in spher_list:
            if mode == 'inference':
                X_ = Model.extract_features(spher, feat_list, mode)
            else:
                X_, y_ = Model.extract_features(spher, feat_list, mode)
                y_list.append(y_)
            X_list.append(X_)

        X = np.concatenate(X_list)
        y = np.concatenate(y_list) if mode != 'inference' else None

        if y is None:
            return X
        else:
            return X, y



    @staticmethod
    def save_model(model, path: str):
        if path is not None:
            with open(path, 'wb') as output:
                pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(path: str):
        with open(path, 'rb') as input:
            return pickle.load(input)

            
class Model_Ki67(Model):

    @staticmethod
    def extract_features(spher, mode='inference'):
        feat_list = [
            lambda blob: blob['features']['signal_filtered_mean'][0],
            lambda blob: blob['features']['nearby_signal_mean'][0],
            lambda blob: blob['features']['signal_mean'][1] # Channel nuclei = 1
        ]
        return Model.extract_features(spher, feat_list, mode)

    @staticmethod
    def extract_features_batch(spher_list: list, mode='inference'):
        feat_list = [
            lambda blob: blob['features']['signal_filtered_mean'][0],
            lambda blob: blob['features']['nearby_signal_mean'][0],
            lambda blob: blob['features']['signal_mean'][1] # Channel nuclei = 1
        ]
        return Model.extract_features_batch(spher_list, feat_list, mode)


class Model_LoD(Model):
    
    @staticmethod
    def extract_features(spher, mode='inference'):
        feat_list = [
            lambda blob: blob['features']['signal_filtered_mean'][0],
            lambda blob: blob['features']['signal_filtered_mean'][2],
        ]
        return Model.extract_features(spher, feat_list, mode)

    @staticmethod
    def extract_features_batch(spher_list: list, mode='inference'):
        feat_list = [
            lambda blob: blob['features']['signal_filtered_mean'][0],
            lambda blob: blob['features']['signal_filtered_mean'][2],
        ]
        return Model.extract_features_batch(spher_list, feat_list, mode)


class Model_Celltracker(Model):
    
    @staticmethod
    def extract_features(spher, mode='inference'):
        feat_list = [
            lambda blob: blob['features']['signal_filtered_mean'][1],
        ]
        return Model.extract_features(spher, feat_list, mode)

    @staticmethod
    def extract_features_batch(spher_list: list, mode='inference'):
        feat_list = [
            lambda blob: blob['features']['signal_filtered_mean'][1],
        ]
        return Model.extract_features_batch(spher_list, feat_list, mode)



def load_spher(path_output):
    with open(path_output, 'rb') as input:
        return pickle.load(input)

def save_spher(f_path, path_output, c_label_image, mode):
    ''' Function to calculate and save the sperhoid object with class labels
    mode: "new" or "old". Defines if the parameters for the new or old data are used
    '''
    
    s = time.time()
    if mode == 'old':
        params = {  'ch_nuclei': 1,
                    'min_volume': None, # in µm^3
                    'max_volume': None, # in µm^3
                    'channel_thresh': [25, 25], # [Ki/Casp, Draq5]
                    'c_label_image': c_label_image,}

    rec = Recording(f_path, ch_nuclei=params['ch_nuclei'])
    spheroid = Spheroid(rec, params)
    spher = spheroid.spher_to_dict()

    if path_output is not None:
        with open(path_output, 'wb') as output:
            pickle.dump(spher, output, pickle.HIGHEST_PROTOCOL)

    d = time.time() - s; print('Duration of calculation: {:.0f} s -> {:.2f} min'.format(d, d/60))

def get_number_of_labels(spheroid_list:list) -> None:
    n_labeled_total = 0
    for spheroid in spheroid_list:
        cells_labeled = [cell['features']['class_label'] for cell in spheroid['Cells'] if cell['features']['class_label']!=0]
        n_labeled_total += len(cells_labeled)
        print(f"Number of labels: {len(cells_labeled)}; number of cells total: {len(spheroid['Cells'])}; Percent labeled: {len(cells_labeled)/len(spheroid['Cells']):.3f}")
        print(f"Label 1: {np.count_nonzero(np.asarray(cells_labeled) == 1)}\nLabel 2: {np.count_nonzero(np.asarray(cells_labeled) == 2)}")

    print(f"Total amount of labeled cells: {n_labeled_total}")
    print("")

def plot_boundary(model, X, y):
    from matplotlib import pyplot as plt
    if X.shape[1] not in [1,2]:
        raise ValueError('Only one or two input dimensions supported.')

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    if X.shape[1] == 1:
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), [0,1])
    elif X.shape[1] == 2:
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if X.shape[1] == 1:
        pred = model.inference(xx.ravel()[:,None])
    if X.shape[1] == 2:
        pred = model.inference(np.stack([xx.ravel(), yy.ravel()], axis=-1))

    # Put the result into a color plot
    pred = pred.reshape(xx.shape)
    plt.contourf(xx, yy, pred, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    if X.shape[1] == 1:
        plt.scatter(X[:, 0], y, c=y, cmap=plt.cm.coolwarm)
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    lim_spacer_x = (xx.max()-xx.min())*0.025
    lim_spacer_y = (yy.max()-yy.min())*0.025
    plt.xlim(xx.min()-lim_spacer_x, xx.max()+lim_spacer_x)
    plt.ylim(yy.min()-lim_spacer_y, yy.max()+lim_spacer_y)
    plt.show()



if __name__ == "__main__":
    from Spheroid_class import Spheroid
    from Recording_class import Recording
    
    ''' Ki67 Model'''
    save_spher('path/to/raw/image.tif', 
              'path/output.data',
              'path/to/class/label/image.tif',
              mode = 'old')
    
    spheroids_train = []
    spheroids_train.append(load_spher('path/output.data'))
    spheroids_test = []
    spheroids_test.append(load_spher('path/output2.data'))
    spheroids_test.append(load_spher('path/output3.data'))

    get_number_of_labels(spheroids_train)
    get_number_of_labels(spheroids_test)

    model_ki67 = Model_Ki67()
    X_train, y_train = Model_Ki67.extract_features_batch(spheroids_train, mode='train')
    model_ki67.train(X_train, y_train)
    X_test, y_test = Model_Ki67.extract_features_batch(spheroids_test, mode='test')
    model_ki67.test(X_test, y_test)

    # plot_boundary(model_ki67, X_test, y_test)

    save_path_model_ki67 = 'path/model/output.data'
    # Model.save_model(model_ki67, save_path_model_ki67)
