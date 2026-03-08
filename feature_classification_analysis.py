import argparse
import json
import math
import re
from collections import Counter

import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from sae import SparseAutoencoder, load_activations

# ----------------------------
# Helper functions
# ----------------------------

def get_node_classes(dataset):
    data = dataset[0]
    return data.y.squeeze(1).to(torch.device("cpu"))


def load_sae(sae_path: str, device: torch.device):
    name = sae_path.split("/")[-1]

    R_match = re.search(r"R(\d+)", name)
    if not R_match:
        raise ValueError(f"Cannot parse R from SAE filename: {name}")
    R = int(R_match.group(1))

    alpha_match = re.search(r"alpha([\d.e-]+)", name)
    if not alpha_match:
        raise ValueError(f"Cannot parse alpha from SAE filename: {name}")
    alpha = float(alpha_match.group(1))

    state_dict = torch.load(sae_path, map_location="cpu")
    d_hid, d_in = state_dict["M"].shape

    sae = SparseAutoencoder(d_in=d_in, R=R, alpha=alpha).to(device)
    sae.load_state_dict(state_dict)
    sae.eval()

    if sae.d_hid != d_hid:
        raise ValueError(f"Loaded SAE d_hid={sae.d_hid} does not match checkpoint d_hid={d_hid}")

    return sae


def compute_purity_entropy(class_counts, K):
    if K == 0:
        return 0.0, 0.0
    majority_count = max(class_counts.values())
    purity = majority_count / K
    entropy = 0.0
    for count in class_counts.values():
        p = count / K
        entropy -= p * math.log(p + 1e-12)
    return purity, entropy


# ----------------------------
# Main feature classification
# ----------------------------

def classify_features_from_activations(
    feature_activations,
    node_classes,
    top_k=20,
    out_path="feature_stats.json",
):
    num_nodes, num_features = feature_activations.shape
    results = {}

    for fid in range(num_features):
        # Top-K nodes for this feature (purely by activation)
        feature_col = feature_activations[:, fid]
        _, indices = torch.topk(feature_col, k=min(top_k, num_nodes))

        class_ids = [int(node_classes[n]) for n in indices]

        counts = dict(Counter(class_ids))
        purity, entropy = compute_purity_entropy(counts, len(indices))

        results[str(fid)] = {
            "purity": purity,
            "entropy": entropy,
            "topk_class_counts": counts
        }

    print(f"Saving results to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")


# ----------------------------
# Script entry point
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze class concentration of trained SAE latent features."
    )
    parser.add_argument(
        "--sae_path",
        type=str,
        default="saes/gcn_l3_h256_r0/sae_layer_0_R2_alpha0.0005_e200_r0.pt",
        help="Path to trained SAE checkpoint.",
    )
    parser.add_argument(
        "--activations_path",
        type=str,
        default="activations/gcn_l3_h256_r0/layer_0_postrelu.pt",
        help="Path to GNN activations [num_nodes, d_in] used as SAE input.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top-activated nodes used per feature.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="feature_stats.json",
        help="Path to write analysis JSON.",
    )
    args = parser.parse_args()

    # Load dataset in sparse format
    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()])
    )
    node_classes = get_node_classes(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_full = load_activations(args.activations_path, device=device)
    sae = load_sae(args.sae_path, device=device)

    with torch.no_grad():
        _, c = sae(x_full)

    feature_activations = c.detach().to(torch.device("cpu"))

    classify_features_from_activations(
        feature_activations=feature_activations,
        node_classes=node_classes,
        top_k=args.top_k,
        out_path=args.out_path,
    )
