import os
import re
import json
import random
import math
import numpy as np

import pandas as pd
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from dotenv import load_dotenv
from openai import OpenAI

from sae import SparseAutoencoder, load_activations

from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import (
    BaseStorage,
    NodeStorage,
    EdgeStorage,
    GlobalStorage,
)

torch.serialization.add_safe_globals([
    Data,
    DataEdgeAttr,
    DataTensorAttr,
    BaseStorage,
    NodeStorage,
    EdgeStorage,
    GlobalStorage,
])

load_dotenv()
client = OpenAI()


# =========================
# SAE Loading
# =========================

def load_sae(sae_path: str, device: torch.device):
    name = os.path.basename(sae_path)

    R = int(re.search(r'R(\d+)', name).group(1))
    alpha = float(re.search(r'alpha([\d.e-]+)', name).group(1))

    state_dict = torch.load(sae_path, map_location='cpu')
    d_hid, d_in = state_dict['M'].shape

    sae = SparseAutoencoder(d_in=d_in, R=R, alpha=alpha).to(device)
    sae.load_state_dict(state_dict)
    sae.eval()

    print(f"Loaded SAE: d_in={d_in}, R={R}, d_hid={sae.d_hid}, alpha={alpha}")
    return sae


# =========================
# Metadata Loading
# =========================

def get_node_metadata(dataset):
    root = dataset.root
    mapping_path = os.path.join(root, "mapping", "nodeidx2paperid.csv.gz")

    node_to_paperid = pd.read_csv(mapping_path, compression="gzip")
    node_to_paperid.columns = ["node_idx", "paper_id"]
    node_to_paperid["node_idx"] = node_to_paperid["node_idx"].astype(int)

    titleabs_df = pd.read_csv(
        "titleabs.tsv.gz",
        sep="\t",
        compression="gzip",
        names=["paper_id", "title", "abstract"],
        header=None,
    )

    node_to_paperid["paper_id"] = node_to_paperid["paper_id"].astype(str)
    titleabs_df["paper_id"] = titleabs_df["paper_id"].astype(str)

    merged_df = node_to_paperid.merge(titleabs_df, on="paper_id", how="left")
    merged_df = merged_df.sort_values("node_idx").reset_index(drop=True)

    return merged_df


# =========================
# SAE activation scaling
# =========================

def rescale_activations(activations: torch.Tensor, max_val: int = 10):
    activations = torch.clamp(activations, min=0)
    max_act = activations.max().item()
    if max_act == 0:
        return torch.zeros_like(activations, dtype=torch.long)
    scaled = (activations / max_act) * max_val
    return scaled.round().long()


# =========================
# LLM YES/NO + LOGPROBS
# =========================

def query_llm_binary(node_info, feature_id, feature_explanation):
    title = node_info.get("title", "")
    abstract = node_info.get("abstract", "")

    prompt = f"""
    You must answer with exactly one word:
    Yes or No.

    Question:
    Is the following feature relevant to the paper?

    Paper Title: {title}
    Paper Abstract: {abstract}

    Feature {feature_id}: {feature_explanation}

    Answer:"""

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1,
        temperature=0,
        logprobs=5,
    )

    choice = response.choices[0]
    text = choice.text.strip()
    top_logprobs = choice.logprobs.top_logprobs[0]

    logprob_yes = top_logprobs.get(" Yes", -100)
    logprob_no = top_logprobs.get(" No", -100)

    prob_yes = math.exp(logprob_yes)
    prob_no = math.exp(logprob_no)

    total = prob_yes + prob_no
    if total > 0:
        prob_yes /= total
        prob_no /= total

    log_odds = logprob_yes - logprob_no

    tokens_used = response.usage.total_tokens if response.usage else None

    return {
        "model_answer": text,
        "prob_yes": prob_yes,
        "prob_no": prob_no,
        "log_odds": log_odds,
        "tokens_used": tokens_used,
    }


# =========================
# New scaling: LLM log-odds → 1-10 based on confidence
# =========================

def scale_llm_log_odds_to_10_confidence(prob_yes):
    return int(round(prob_yes * 10))


# =========================
# Validation Loop with extra low-activation features
# =========================

def validate_nodes(
    sae,
    x_full,
    node_metadata,
    feature_explanations,
    out_path,
    num_nodes=50,
    topk=3,
):
    device = next(sae.parameters()).device

    with torch.no_grad():
        _, c = sae(x_full.to(device))

    num_nodes_total = c.shape[0]
    selected_nodes = random.sample(
        range(num_nodes_total),
        min(num_nodes_total, num_nodes),
    )

    results = []

    for node_idx in selected_nodes:
        node_info = node_metadata.iloc[node_idx].to_dict()

        # Get top 100 features first
        top100_vals, top100_inds = torch.topk(c[node_idx], k=100)
        top100_vals = top100_vals.cpu()
        top100_inds = top100_inds.cpu()

        # First 3 are original top-k
        top_vals = top100_vals[:topk]
        top_inds = top100_inds[:topk]

        # Remaining 97 for low-activation candidates
        low_candidate_vals = top100_vals[topk:]
        low_candidate_inds = top100_inds[topk:]

        # Pick 2 lowest-activated features among candidates
        if len(low_candidate_vals) >= 2:
            low_indices_sorted = torch.argsort(low_candidate_vals)[:2]
            low_vals = low_candidate_vals[low_indices_sorted]
            low_inds = low_candidate_inds[low_indices_sorted]
        else:
            low_vals = low_candidate_vals
            low_inds = low_candidate_inds

        # Combine top-k + 2 low
        feature_vals = torch.cat([top_vals, low_vals])
        feature_inds = torch.cat([top_inds, low_inds])

        raw_vals = feature_vals.tolist()
        scaled_vals = rescale_activations(feature_vals)

        llm_results = []

        entry = {
            "node_id": node_idx,
            "paper_title": node_info.get("title", ""),
            "top_features": [],
        }

        for i, fid in enumerate(feature_inds):
            fid_str = str(fid.item())
            explanation = feature_explanations.get(fid_str, "No explanation available.")

            llm_result = query_llm_binary(
                node_info,
                fid,
                explanation,
            )
            llm_results.append(llm_result)

            llm_scaled_conf = scale_llm_log_odds_to_10_confidence(llm_result["prob_yes"])

            entry["top_features"].append({
                "feature_id": fid.item(),
                "true_activation_raw": raw_vals[i],
                "true_activation_scaled_1_10": int(scaled_vals[i]),
                "llm_answer": llm_result["model_answer"],
                "llm_scaled_score_1_10": llm_scaled_conf,
                "prob_yes": llm_result["prob_yes"],
                "prob_no": llm_result["prob_no"],
                "log_odds": llm_result["log_odds"],
                "tokens_used": llm_result["tokens_used"],
            })

            print(
                f"Node {node_idx} | Feature {fid.item()} | "
                f"SAE scaled={scaled_vals[i]} | "
                f"LLM scaled={llm_scaled_conf}"
            )

        results.append(entry)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("Validation complete. Saved to", out_path)
    return results


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    sae_path = "saes/gcn_l3_h256_r0/sae_layer_1_R2_alpha0.005_e200_r0.pt"
    acts_path = "activations/gcn_l3_h256_r0/layer_1_postrelu.pt"
    feature_expl_path = "feature_explanations/feature_explanations_long.json"

    out_path = "node_feature_validation_binary_confidence_extra.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]),
    )

    node_metadata = get_node_metadata(dataset)
    x_full = load_activations(acts_path, device=device)
    sae = load_sae(sae_path, device=device)

    with open(feature_expl_path, "r") as f:
        raw = json.load(f)
        feature_explanations = {
            k: v["explanation"] for k, v in raw.items()
        }

    validate_nodes(
        sae=sae,
        x_full=x_full,
        node_metadata=node_metadata,
        feature_explanations=feature_explanations,
        out_path=out_path,
        num_nodes=10,
        topk=3,
    )