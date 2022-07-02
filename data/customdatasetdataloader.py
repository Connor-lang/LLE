import torch
from data.unaligned_dataset import UnalignedDataset
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = UnalignedDataset()
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
            
    def load_data(self):
        return self.dataloader
    
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)