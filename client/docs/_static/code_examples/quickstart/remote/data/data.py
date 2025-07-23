from ML_management.dataset_loader.dataset_loader_pattern import DatasetLoaderPattern


class EmptyDataWrapper(DatasetLoaderPattern):
    """Empty dataset loader class."""

    def get_dataset(self):
        """Return None."""
        return None
