import os.path as osp

import argparse
import jsonlines
import torch
import scipy.sparse as sp
from torch_sparse import coalesce
from torch_geometric.datasets import Reddit
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler, Data
from torch_geometric.nn import SAGEConv, GATConv

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAT')
parser.add_argument('--edge_path', type=str)
parser.add_argument('--node_path', type=str)
args = parser.parse_args()
assert args.model in ['SAGE', 'GAT']

def get_vocabulary(node_path):
    paper_ids = set()
    for line in jsonlines.open(node_path):
        paper_ids.add(line['paper_id'])
    return {paper_id:i for i, paper_id in enumerate(paper_ids)}

def load_edge_index(edge_path, vocabulary):

    row_ind = []
    col_ind = []

    for edge in jsonlines.open(edge_path):
        citing_idx = vocabulary.get(edge['metadata']['citing_paper'])
        cited_idx = vocabulary.get(edge['metadata']['cited_paper'])
        if cited_idx and citing_idx:
            row_ind.append(citing_idx)
            col_ind.append(cited_idx)
    row = torch.tensor(row_ind, dtype=torch.long)
    col = torch.tensor(col_ind, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    #edge_index, _ = coalesce(edge_index, None, len(vocabulary), len(vocabulary))
    return edge_index

class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, concat=False):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16, normalize=False, concat=concat)
        self.conv2 = SAGEConv(16, out_channels, normalize=False, concat=concat)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(
            self.conv1((x, None), data.edge_index, size=data.size,
                       res_n_id=data.res_n_id))
        x = F.dropout(x, p=0.5, training=self.training)
        data = data_flow[1]
        x = self.conv2((x, None), data.edge_index, size=data.size,
                       res_n_id=data.res_n_id)
        return F.log_softmax(x, dim=1)

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=True,
                             dropout=0.6)

    def forward(self, x, data_flow):
        block = data_flow[0]
        x = x[block.n_id]
        x = F.elu(
            self.conv1((x, x[block.res_n_id]), block.edge_index,
                       size=block.size))
        x = F.dropout(x, p=0.6, training=self.training)
        block = data_flow[1]
        x = self.conv2((x, x[block.res_n_id]), block.edge_index,
                       size=block.size)
        return F.log_softmax(x, dim=1)

vocabulary = get_vocabulary(args.node_path)
edge_index = load_edge_index(args.edge_path, vocabulary)

data = Data(torch.ones(len(vocabulary), 100, dtype=torch.float), edge_index)
data.y = torch.ones(len(vocabulary), dtype=torch.long)
data.train_mask = torch.ones(len(vocabulary), dtype=torch.long)
data.test_mask = torch.ones(len(vocabulary), dtype=torch.long)
data.val_mask = torch.ones(len(vocabulary), dtype=torch.long)

loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=1000,
                                 shuffle=True, add_self_loops=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Net = SAGENet if args.model == 'SAGE' else GATNet
model = Net(100, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()

    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()


def test(mask):
    model.eval()

    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device)).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
    return correct / mask.sum().item()


for epoch in range(1, 31):
    loss = train()
    test_acc = test(data.test_mask)
    print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
