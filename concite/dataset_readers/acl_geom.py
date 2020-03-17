import os.path as osp

import torch
from torch_geometric.data import Dataset

class ACLGeom(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ACLGeom, self).__init__(root, transform, pre_transform)

    @property
    def file_name(self):
        return ['data_1.jsonl', 'data_2.jsonl', 'data_3.jsonl', 'data_4.jsonl']

    def process(self):
        data = Data(...)
        if self.pre_filter is not None and not self.pre_filter(data):
            continue

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
