import torch
import os
import re
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class GCN_Frozen(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN_Frozen, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

def parse_ckpt(ckpt_path):
    name = os.path.basename(ckpt_path)

    pattern = r"gcn_l(?P<num_layers>\d+)_h(?P<hidden_channels>\d+)_e(?P<epochs>\d+)_r(?P<run>\d+)\.pt"
    match = re.match(pattern, name)

    if match is None:
        raise ValueError(f"Checkpoint name does not match expected format: {name}")

    cfg = {k: int(v) for k, v in match.groupdict().items()}
    return cfg

def load_gcn(ckpt_path, dropout=0.5):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(),T.ToSparseTensor()]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = dataset[0]
    data = data.to(device)

    cfg = parse_ckpt(ckpt_path)

    model = GCN_Frozen(
        in_channels=data.num_features,                
        hidden_channels=cfg["hidden_channels"],
        out_channels=dataset.num_classes, 
        num_layers=cfg["num_layers"],
        dropout=dropout                        
    )
    
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model

if __name__ == "__main__":
    ckpt_path = "models/gcn_l3_h256_e500_r0.pt"
    load_gcn(ckpt_path)    
