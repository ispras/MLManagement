from ML_management.dataset_loader.dataset_loader_pattern import DatasetLoaderPattern


class DataWrapper(DatasetLoaderPattern):

    def get_dataset(self, ...):
        
        data = ...  # read raw data from self.data_path

        return data
