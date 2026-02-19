import os
import re
import json
import random


import pandas as pd
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from dotenv import load_dotenv
from openai import OpenAI

from sae import SparseAutoencoder, load_activations

import torch
# from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# torch.serialization.add_safe_globals([
#     DataEdgeAttr,
#     DataTensorAttr,
# ])

load_dotenv()
client = OpenAI()

def load_sae(sae_path: str, device: torch.device):
    """Load SAE and parse R/alpha from filename"""
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
    """Load node->paper mapping + titles/abstracts"""
    root = dataset.root
    mapping_path = os.path.join(root, "mapping", "nodeidx2paperid.csv.gz")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Cannot find mapping file at {mapping_path}")
    
    node_to_paperid = pd.read_csv(mapping_path, compression="gzip")
    if "node idx" in node_to_paperid.columns and "paper id" in node_to_paperid.columns:
        node_to_paperid = node_to_paperid.rename(columns={"node idx": "node_idx", "paper id": "paper_id"})
    else:
        node_to_paperid = node_to_paperid.iloc[:, :2].copy()
        node_to_paperid.columns = ["node_idx", "paper_id"]
    node_to_paperid["node_idx"] = pd.to_numeric(node_to_paperid["node_idx"], errors="coerce").astype(int)
    node_to_paperid = node_to_paperid.drop_duplicates(subset=["node_idx"]).sort_values("node_idx").reset_index(drop=True)

    titleabs_path = "titleabs.tsv.gz"
    if not os.path.exists(titleabs_path):
        raise FileNotFoundError(f"Cannot find {titleabs_path}")
    titleabs_df = pd.read_csv(titleabs_path, sep="\t", compression="gzip",
                              names=["paper_id", "title", "abstract"], header=None)

    node_to_paperid["paper_id"] = node_to_paperid["paper_id"].astype(str)
    titleabs_df["paper_id"] = titleabs_df["paper_id"].astype(str)

    merged_df = node_to_paperid.merge(titleabs_df, on="paper_id", how="left")
    merged_df = merged_df.sort_values("node_idx").reset_index(drop=True)
    return merged_df

def rescale_activations(activations: torch.Tensor, max_val: int = 10):
    activations = torch.clamp(activations, min=0)
    max_act = activations.max().item()
    if max_act == 0:
        return torch.zeros_like(activations, dtype=torch.long)
    scaled = (activations / max_act) * max_val
    return scaled.round().long()

def query_llm(node_info, top_features, feature_explanations):
    """Send node + top feature info to LLM and get response, returning usage"""
    node_id = node_info['node_idx']
    title = node_info.get('title', '')
    abstract = node_info.get('abstract', '')

    feature_lines = []
    for fid, act_val in zip(top_features['ids'], top_features['vals']):
        expl = feature_explanations.get(str(fid), "No explanation available.")
        feature_lines.append(f"- Feature {fid} (activation {act_val:.2f}): {expl}")

    prompt = (
        f"You are validating a sparse autoencoder feature.\n\n"
        f"Node:\n- ID: {node_id}\n- Title: {title}\n- Abstract: {abstract}\n\n"
        f"Activated Features:\n" + "\n".join(feature_lines) +
        "\n\nQuestion: Does it make sense that these features strongly activate for this node? Explain briefly."
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": prompt}],
    )

    # Add token usage tracking
    usage = getattr(resp, "usage", None)
    if usage:
        total_tokens = usage.total_tokens
    else:
        total_tokens = None

    return resp.output_text.strip(), total_tokens

def validate_nodes(sae, x_full, node_metadata, feature_explanations, out_path, num_nodes=50, topk=3):
    device = next(sae.parameters()).device
    with torch.no_grad():
        _, c = sae(x_full.to(device))
    
    num_nodes_total = c.shape[0]
    selected_nodes = random.sample(range(num_nodes_total), min(num_nodes_total, num_nodes))

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
    else:
        results = []

    for node_idx in selected_nodes:
        node_info = node_metadata.iloc[node_idx].to_dict()
        top_inds, top_vals = torch.topk(c[node_idx], k=topk)
        top_features = {'ids': top_inds.tolist(), 'vals': top_vals.tolist()}

        llm_resp, tokens_used = query_llm(node_info, top_features, feature_explanations)

        entry = {
            "node_id": node_idx,
            "top_features": top_features,
            "llm_response": llm_resp,
            "tokens_used": tokens_used
        }
        results.append(entry)

        # Save incrementally in readable JSON
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Processed node {node_idx} → {tokens_used} tokens used, LLM response saved.")

    print("Validation complete. JSON saved to", out_path)
    return results

if __name__ == "__main__":
    sae_path = "saes/gcn_l3_h256_r0/sae_layer_1_R2_alpha0.005_e200_r0.pt"
    acts_path = "activations/gcn_l3_h256_r0/layer_1_postrelu.pt"
    feature_expl_path = "feature_explanations.json"
    out_path = "node_feature_validation.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()])
    )

    node_metadata = get_node_metadata(dataset)
    x_full = load_activations(acts_path, device=device)
    sae = load_sae(sae_path, device=device)

    # Load previous feature explanations for LLM context
    if os.path.exists(feature_expl_path):
        with open(feature_expl_path, "r") as f:
            feature_explanations = {k: v["explanation"] for k, v in json.load(f).items()}
    else:
        feature_explanations = {}

    validate_nodes(
        sae=sae,
        x_full=x_full,
        node_metadata=node_metadata,
        feature_explanations=feature_explanations,
        out_path=out_path,
        num_nodes=10,
        topk=3,
    )