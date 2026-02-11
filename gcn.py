import torch
import os
import re
from gnn import GCN


def parse_ckpt(ckpt_path):
    name = os.path.basename(ckpt_path)

    pattern = r"gcn_l(?P<num_layers>\d+)_h(?P<hidden_channels>\d+)_e(?P<epochs>\d+)_r(?P<run>\d+)\.pt"
    match = re.match(pattern, name)

    if match is None:
        raise ValueError(f"Checkpoint name does not match expected format: {name}")

    cfg = {k: int(v) for k, v in match.groupdict().items()}
    return cfg

def load_gcn(ckpt_path, data, dropout=0.5):
    cfg = parse_ckpt(ckpt_path)

    model = GCN(
        in_channels=data.num_features,                
        hidden_channels=cfg["hidden_channels"],
        out_channels=data.num_classes,              
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
    ckpt_path = "gcn_l13_h256333_e5030_r12.pt"
    cfg = parse_ckpt(ckpt_path)
    print(cfg)



