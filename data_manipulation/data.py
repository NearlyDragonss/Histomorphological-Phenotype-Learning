import os
from data_manipulation.dataset import Dataset
from data_manipulation.datasetPyTorch import DatasetPyTorch


class Data:
    def __init__(self, dataset, marker, patch_h, patch_w, n_channels, batch_size, project_path=os.getcwd(), thresholds=(), labels=None, empty=False, num_clusters=500, clust_percent=1.0, load=True):
        # Directories and file name handling.
        self.dataset = dataset
        self.marker = marker
        self.dataset_name = '%s_%s' % (self.dataset, self.marker)
        relative_dataset_path = os.path.join(self.dataset, self.marker)
        relative_dataset_path = os.path.join('datasets', relative_dataset_path)
        relative_dataset_path = os.path.join(project_path, relative_dataset_path)
        self.pathes_path = os.path.join(relative_dataset_path, 'patches_h%s_w%s' % (patch_h, patch_w))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels
        self.batch_size = batch_size

        # Train dataset
        self.hdf5_train = os.path.join(self.pathes_path, 'hdf5_%s_train.h5' % self.dataset_name)
        print('Train Set:', self.hdf5_train)
        self.training = None
        if os.path.isfile(self.hdf5_train) and load:
            self.training = DatasetPyTorch(self.hdf5_train, patch_h, patch_w, n_channels, batch_size=batch_size, thresholds=thresholds, labels=labels, empty=empty, num_clusters=num_clusters, clust_percent=clust_percent)

        # Validation dataset, some datasets work with those.
        self.hdf5_validation = os.path.join(self.pathes_path, 'hdf5_%s_validation.h5' % self.dataset_name)
        print('Validation Set:', self.hdf5_validation)
        self.validation = None
        if os.path.isfile(self.hdf5_validation) and load:
            self.validation = DatasetPyTorch(self.hdf5_validation, patch_h, patch_w, n_channels, batch_size=batch_size, thresholds=thresholds, labels=None, empty=empty)

        # Test dataset
        self.hdf5_test = os.path.join(self.pathes_path, 'hdf5_%s_test.h5' % self.dataset_name)
        print('Test Set:', self.hdf5_test)
        self.test = None
        if os.path.isfile(self.hdf5_test) and load:
            self.test = DatasetPyTorch(self.hdf5_test, patch_h, patch_w, n_channels, batch_size=batch_size, thresholds=thresholds, labels=None, empty=empty)
        print()
