import os
import re
import json

import torch
import pandas as pd

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from dotenv import load_dotenv
from openai import OpenAI

from sae import SparseAutoencoder, load_activations


load_dotenv()
client = OpenAI()


def load_sae(sae_path: str, device: torch.device):
    name = os.path.basename(sae_path)
    
    R_match = re.search(r'R(\d+)', name)
    if not R_match:
        raise ValueError(f"Cannot parse R from: {name}")
    R = int(R_match.group(1))
    
    alpha_match = re.search(r'alpha([\d.e-]+)', name)
    if not alpha_match:
        raise ValueError(f"Cannot parse alpha from: {name}")
    alpha = float(alpha_match.group(1))
    
    state_dict = torch.load(sae_path, map_location='cpu')
    d_hid, d_in = state_dict['M'].shape

    sae = SparseAutoencoder(d_in=d_in, R=R, alpha=alpha).to(device)
    sae.load_state_dict(state_dict)
    sae.eval()
    
    print(f"Loaded SAE: d_in={d_in}, R={R}, d_hid={sae.d_hid}, alpha={alpha}")
    return sae


def get_node_metadata(dataset):
    root = dataset.root

    mapping_path = os.path.join(root, "mapping", "nodeidx2paperid.csv.gz")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Could not find mapping file at {mapping_path}")

    print(f"Loading node to paper ID mapping from {mapping_path}...")
    node_to_paperid = pd.read_csv(mapping_path, compression="gzip")

    if "node idx" in node_to_paperid.columns and "paper id" in node_to_paperid.columns:
        node_to_paperid = node_to_paperid.rename(
            columns={"node idx": "node_idx", "paper id": "paper_id"}
        )
    else:
        node_to_paperid = node_to_paperid.iloc[:, :2].copy()
        node_to_paperid.columns = ["node_idx", "paper_id"]

    node_to_paperid["node_idx"] = pd.to_numeric(
        node_to_paperid["node_idx"], errors="coerce"
    )
    node_to_paperid = node_to_paperid.dropna(subset=["node_idx"])
    node_to_paperid["node_idx"] = node_to_paperid["node_idx"].astype(int)

    node_to_paperid = (
        node_to_paperid
        .drop_duplicates(subset=["node_idx"])
        .sort_values("node_idx")
        .reset_index(drop=True)
    )

    print(f"  Loaded {len(node_to_paperid)} node mappings")

    titleabs_path = "titleabs.tsv.gz"
    if not os.path.exists(titleabs_path):
        raise FileNotFoundError(f"Could not find {titleabs_path}")

    print(f"Loading paper titles and abstracts from {titleabs_path}...")
    titleabs_df = pd.read_csv(
        titleabs_path,
        sep="\t",
        compression="gzip",
        names=["paper_id", "title", "abstract"],
        header=None,
    )
    print(f"  Loaded {len(titleabs_df)} paper titles/abstracts")

    node_to_paperid["paper_id"] = node_to_paperid["paper_id"].astype(str)
    titleabs_df["paper_id"] = titleabs_df["paper_id"].astype(str)

    merged_df = node_to_paperid.merge(titleabs_df, on="paper_id", how="left")
    merged_df = merged_df.sort_values("node_idx").reset_index(drop=True)

    print(f"  Merged dataset: {len(merged_df)} entries")
    print(f"  Entries with titles: {merged_df['title'].notna().sum()}")

    return merged_df


def rescale_activations(activations: torch.Tensor, max_val: int = 10):
    activations = torch.clamp(activations, min=0)

    max_act = activations.max().item()

    if max_act == 0:
        return torch.zeros_like(activations, dtype=torch.long)

    scaled = (activations / max_act) * max_val
    return scaled.round()


def topk_nodes_for_feature(c: torch.Tensor, feature_id: int, node_metadata, freq: torch.Tensor, K: int = 10):
    scores = c[:, feature_id].float()
    adjusted = scores / torch.log1p(freq + 1e-6) ** 0.5

    values, indices = torch.topk(adjusted, k=K)
    scaled = rescale_activations(values, max_val=10)

    examples_list = []
    lines = []
    for node_idx, score in zip(indices.tolist(), scaled.tolist()):
        row = node_metadata.iloc[node_idx]
        title = "" if pd.isna(row["title"]) else str(row["title"])
        abstract = "" if pd.isna(row["abstract"]) else str(row["abstract"])

        examples_list.append({
            "node_idx": int(node_idx),
            "score_0_10": int(score),
            "title": title,
            "abstract": abstract,
        })

        lines.append(f"- ({score}) {title}\n  {abstract}")

    examples_text = "\n".join(lines)
    return examples_text, examples_list


def compute_node_frequency(c: torch.Tensor, K: int = 20):
    device = c.device
    num_nodes = c.shape[0]

    freq = torch.zeros(num_nodes, device=device)

    for fid in range(c.shape[1]):
        col = c[:, fid]

        if col.max().item() == 0:
            continue

        _, idx = torch.topk(col, K)
        freq[idx] += 1.0

    return freq


def interpret_feature(feature_id: int, examples_text: str):
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are labeling a latent feature in a neural network. "
                    "Given examples, produce a short, concrete topic label. "
                    "Write ONE concise sentence fragment. "
                    "No filler. No explanations. "
                    "Do NOT say 'this feature' or 'this latent feature'. "
                    "Max 12 words."
                ),
            },
            {
                "role": "user",
                "content": f"Top activating papers:\n{examples_text}\n\nLabel:",
            },
        ],
    )
    return resp.output_text.strip()


def interpret_features(
    sae,
    x_full,
    node_metadata,
    out_path: str,
    K: int = 5,
):
    device = next(sae.parameters()).device

    with torch.no_grad():
        _, c = sae(x_full.to(device))

    print("c shape:", tuple(c.shape))

    print("Computing node frequencies...")
    freq = compute_node_frequency(c, K=20)

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} features already done.")
    else:
        results = {}

    num_features = c.shape[1]

    for fid in range(num_features):
        key = str(fid)

        if key in results:
            continue

        if c[:, fid].max().item() == 0:
            continue

        print(f"Interpreting feature {fid}...")

        examples_text, examples_list = topk_nodes_for_feature(c, fid, node_metadata, freq, K=K)
        explanation = interpret_feature(fid, examples_text)

        results[key] = {
            "explanation": explanation,
            "topk": examples_list,
        }

        print(f"Feature {fid}: {explanation}")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("Done. Results saved to", out_path)
    return results


if __name__ == "__main__":
    sae_path = "saes/gcn_l3_h256_r0/sae_layer_1_R2_alpha0.005_e200_r0.pt"
    acts_path = "activations/gcn_l3_h256_r0/layer_1_postrelu.pt"
    out_path = "feature_explanations.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()])
    )

    node_metadata = get_node_metadata(dataset)
    x_full = load_activations(acts_path, device=device)
    sae = load_sae(sae_path, device=device)

    interpret_features(
        sae=sae,
        x_full=x_full,
        node_metadata=node_metadata,
        out_path=out_path,
        K=10,
    )
