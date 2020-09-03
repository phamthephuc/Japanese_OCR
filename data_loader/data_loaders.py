from base import BaseDataLoader
from data_set import data_sets

from utils import full2half

class TextImageDataLoader(BaseDataLoader):
    """
    TextImageDataLoader data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, imgH = 32, imgW = 100, keep_ratio = False, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        target_transform = full2half
        self.dataset = data_sets.ImageTextDataset(data_dir, target_transform)
        collate_fn = data_sets.alignCollate(imgH, imgW, keep_ratio)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)



class TextImageDataLoaderForTest(BaseDataLoader):
    """
    TextImageDataLoader data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, imgH = 32, imgW = 100, keep_ratio = False, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        target_transform = full2half
        self.dataset = data_sets.ImageTextDatasetForTest(data_dir, None, target_transform)
        collate_fn = data_sets.alignCollate(imgH, imgW, keep_ratio)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)