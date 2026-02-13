import os
import re
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    def forward(self, x, adj_t, act_dir):
        os.makedirs(act_dir, exist_ok=True)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)

            torch.save(
                x.detach().cpu(),
                os.path.join(act_dir, f"layer_{i}_postrelu.pt")
            )

        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def parse_ckpt(ckpt_path):
    name = os.path.basename(ckpt_path)
    match = re.search(r"gcn_l(\d+)_h(\d+)", name)

    if match is None:
        raise ValueError(f"Invalid checkpoint name: {name}")

    num_layers = int(match.group(1))
    hidden_channels = int(match.group(2))

    return num_layers, hidden_channels


def activation_save_path(ckpt_path, base_dir="activations"):
    os.makedirs(base_dir, exist_ok=True)

    num_layers, hidden_channels = parse_ckpt(ckpt_path)
    run = 0

    while True:
        name = f"gcn_l{num_layers}_h{hidden_channels}_r{run}"
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            return path
        run += 1


def load_gcn(ckpt_path):
    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = dataset[0].to(device)

    num_layers, hidden_channels = parse_ckpt(ckpt_path)

    model = GCN(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=num_layers,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, data


if __name__ == "__main__":
    ckpt_path = "models/gcn_l3_h512_e500_r0.pt"
    model, data = load_gcn(ckpt_path)
    act_dir = activation_save_path(ckpt_path)

    with torch.no_grad():
        out = model(data.x, data.adj_t, act_dir)
