import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from gcn import load_gcn
from sae import SparseAutoencoder


GCN_CKPT_PATH = "models/gcn_l3_h256_e500_r0.pt"
SAE_CKPT_PATH = "saes/gcn_l3_h256_r0/sae_layer_0_R2_alpha0.0005_e200_r0.pt"
FEATURE_EXPLANATIONS_PATH = "feature_explanations_layer0_low_alpha.json"
FEATURE_STATS_PATH = "feature_stats.json"


def _load_feature_explanations(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = str(v.get("explanation", ""))
        else:
            out[str(k)] = str(v)
    return out


def _load_feature_stats(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_sae(sae_path: str, device: torch.device) -> SparseAutoencoder:
    name = os.path.basename(sae_path)
    r_match = re.search(r"R(\d+)", name)
    a_match = re.search(r"alpha([\d.e-]+)", name)
    if not r_match or not a_match:
        raise ValueError(f"Cannot parse SAE hyperparameters from: {name}")

    R = int(r_match.group(1))
    alpha = float(a_match.group(1))

    state_dict = torch.load(sae_path, map_location="cpu")
    d_hid, d_in = state_dict["M"].shape
    sae = SparseAutoencoder(d_in=d_in, R=R, alpha=alpha).to(device)
    sae.load_state_dict(state_dict)
    sae.eval()
    if sae.d_hid != d_hid:
        raise ValueError(f"SAE mismatch: expected d_hid={d_hid}, got {sae.d_hid}")
    return sae


def _parse_layer_from_sae(sae_path: str) -> int:
    name = os.path.basename(sae_path)
    m = re.search(r"sae_layer_(\d+)_", name)
    if not m:
        raise ValueError(f"Could not parse layer from SAE filename: {name}")
    return int(m.group(1))


@torch.no_grad()
def _forward_baseline(model, data) -> torch.Tensor:
    x = data.x
    for i, conv in enumerate(model.convs[:-1]):
        x = conv(x, data.adj_t)
        x = model.bns[i](x)
        x = F.relu(x)
    x = model.convs[-1](x, data.adj_t)
    return x.log_softmax(dim=-1)


@torch.no_grad()
def _forward_with_steering(
    model,
    data,
    sae: SparseAutoencoder,
    steer_layer: int,
    feature_ids: List[int],
    mode: str,
    strength: float,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    x = data.x
    for i, conv in enumerate(model.convs[:-1]):
        x = conv(x, data.adj_t)
        x = model.bns[i](x)
        x = F.relu(x)

        if i == steer_layer:
            c = F.relu(x @ sae.M.t() + sae.b)
            feat_idx = torch.tensor(feature_ids, device=x.device, dtype=torch.long)
            feats = c[:, feat_idx]
            dirs = sae.M[feat_idx]

            if mode == "ablate":
                delta = -feats
            elif mode == "boost":
                scales = torch.quantile(feats, q=0.9, dim=0)
                means = feats.mean(dim=0) + 1e-6
                scales = torch.where(scales > 1e-8, scales, means)
                delta = torch.ones_like(feats) * (strength * scales.unsqueeze(0))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            delta = delta * target_mask.float().unsqueeze(1)
            x = x + delta @ dirs
            x = torch.clamp(x, min=0.0)

    x = model.convs[-1](x, data.adj_t)
    return x.log_softmax(dim=-1)


def _pick_features_for_class(
    class_id: int,
    feature_stats: Dict[str, dict],
    min_purity: float,
    min_majority_count: int,
) -> Tuple[List[int], str]:
    strong = []
    weak = []

    for fid_str, stats in feature_stats.items():
        counts = stats.get("topk_class_counts", {})
        counts_int = {int(k): int(v) for k, v in counts.items()}
        if not counts_int:
            continue

        majority_class, majority_count = max(counts_int.items(), key=lambda x: x[1])
        class_count = counts_int.get(class_id, 0)
        purity = float(stats.get("purity", 0.0))

        if majority_class == class_id and purity >= min_purity and majority_count >= min_majority_count:
            strong.append((purity, majority_count, class_count, int(fid_str)))
        elif class_count > 0:
            weak.append((class_count, purity, majority_count, int(fid_str)))

    if strong:
        strong.sort(reverse=True)
        return [item[3] for item in strong], "strong"
    if weak:
        weak.sort(reverse=True)
        return [item[3] for item in weak], "fallback"
    return [], "none"


def _summarize_effects(
    baseline_logp: torch.Tensor,
    steered_logp: torch.Tensor,
    eval_mask: torch.Tensor,
    class_mask_true: torch.Tensor,
    class_id: int,
) -> dict:
    base_pred = baseline_logp.argmax(dim=-1)
    steer_pred = steered_logp.argmax(dim=-1)

    eval_idx = eval_mask.nonzero(as_tuple=True)[0]
    cls_idx = class_mask_true.nonzero(as_tuple=True)[0]

    if cls_idx.numel() == 0:
        return {
            "num_class_nodes": 0,
            "prediction_flip_rate": None,
            "mean_delta_logprob_target_class": None,
            "mean_delta_prob_target_class": None,
            "target_classification_rate_before": None,
            "target_classification_rate_after": None,
            "pred_count_target_before": int((base_pred[eval_idx] == class_id).sum().item()),
            "pred_count_target_after": int((steer_pred[eval_idx] == class_id).sum().item()),
        }

    delta_logp = steered_logp[cls_idx, class_id] - baseline_logp[cls_idx, class_id]
    base_prob = baseline_logp[cls_idx, class_id].exp()
    steer_prob = steered_logp[cls_idx, class_id].exp()
    flip_rate = (base_pred[cls_idx] != steer_pred[cls_idx]).float().mean().item()

    target_rate_before = (base_pred[cls_idx] == class_id).float().mean().item()
    target_rate_after = (steer_pred[cls_idx] == class_id).float().mean().item()

    return {
        "num_class_nodes": int(cls_idx.numel()),
        "prediction_flip_rate": float(flip_rate),
        "mean_delta_logprob_target_class": float(delta_logp.mean().item()),
        "mean_delta_prob_target_class": float((steer_prob - base_prob).mean().item()),
        "target_classification_rate_before": float(target_rate_before),
        "target_classification_rate_after": float(target_rate_after),
        "pred_count_target_before": int((base_pred[eval_idx] == class_id).sum().item()),
        "pred_count_target_after": int((steer_pred[eval_idx] == class_id).sum().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Class-wise latent steering with GCN + SAE.")
    parser.add_argument("--min_purity", type=float, default=0.55)
    parser.add_argument("--min_majority_count", type=int, default=8)
    parser.add_argument("--boost_strength", type=float, default=1.0)
    parser.add_argument("--top_k_features", type=int, default=3)
    parser.add_argument(
        "--all_nodes",
        action="store_true",
        help="Evaluate and steer over all nodes instead of test split only.",
    )
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    gcn_ckpt = GCN_CKPT_PATH
    sae_ckpt = SAE_CKPT_PATH
    expl_path = FEATURE_EXPLANATIONS_PATH
    stats_path = FEATURE_STATS_PATH

    if not os.path.exists(gcn_ckpt):
        raise FileNotFoundError(f"Missing GCN checkpoint: {gcn_ckpt}")
    if not os.path.exists(sae_ckpt):
        raise FileNotFoundError(f"Missing SAE checkpoint: {sae_ckpt}")
    if not os.path.exists(expl_path):
        raise FileNotFoundError(f"Missing feature explanations file: {expl_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing feature stats file: {stats_path}")

    out_path = args.out_path or f"steering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"Using GCN checkpoint: {gcn_ckpt}")
    print(f"Using SAE checkpoint: {sae_ckpt}")
    print(f"Using feature explanations: {expl_path}")
    print(f"Using feature stats: {stats_path}")

    feature_explanations = _load_feature_explanations(expl_path)
    feature_stats = _load_feature_stats(stats_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = load_gcn(gcn_ckpt)
    sae = _load_sae(sae_ckpt, device=device)
    steer_layer = _parse_layer_from_sae(sae_ckpt)

    if steer_layer >= len(model.convs) - 1:
        raise ValueError(
            f"SAE layer {steer_layer} is invalid for model with {len(model.convs) - 1} hidden layers."
        )

    labels = data.y.squeeze(1)
    num_classes = int(labels.max().item()) + 1

    eval_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    if args.all_nodes:
        eval_mask[:] = True
    else:
        from ogb.nodeproppred import PygNodePropPredDataset
        import torch_geometric.transforms as T

        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv",
            transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]),
        )
        split_idx = dataset.get_idx_split()
        eval_mask[split_idx["test"].to(device)] = True

    with torch.no_grad():
        baseline_logp = _forward_baseline(model, data)
    baseline_pred = baseline_logp.argmax(dim=-1)

    class_results = []
    for class_id in range(num_classes):
        class_mask_true = (labels == class_id) & eval_mask
        class_mask_pred = (baseline_pred == class_id) & eval_mask
        class_support = int(class_mask_true.sum().item())

        ranked_fids, match_type = _pick_features_for_class(
            class_id=class_id,
            feature_stats=feature_stats,
            min_purity=args.min_purity,
            min_majority_count=args.min_majority_count,
        )
        selected_fids = ranked_fids[: max(1, args.top_k_features)]

        record = {
            "class_id": class_id,
            "class_support_eval_nodes": class_support,
            "steering_target_nodes_predicted_class": int(class_mask_pred.sum().item()),
            "selected_feature_id": selected_fids[0] if selected_fids else None,
            "selected_feature_ids": selected_fids,
            "match_type": match_type,
            "feature_explanations": [feature_explanations.get(str(fid), "") for fid in selected_fids],
            "ablate": None,
            "boost": None,
        }

        if len(selected_fids) == 0 or class_support == 0 or int(class_mask_pred.sum().item()) == 0:
            class_results.append(record)
            continue

        with torch.no_grad():
            ablate_logp = _forward_with_steering(
                model=model,
                data=data,
                sae=sae,
                steer_layer=steer_layer,
                feature_ids=selected_fids,
                mode="ablate",
                strength=args.boost_strength,
                target_mask=class_mask_pred,
            )
            boost_logp = _forward_with_steering(
                model=model,
                data=data,
                sae=sae,
                steer_layer=steer_layer,
                feature_ids=selected_fids,
                mode="boost",
                strength=args.boost_strength,
                target_mask=class_mask_pred,
            )

        record["ablate"] = _summarize_effects(
            baseline_logp=baseline_logp,
            steered_logp=ablate_logp,
            eval_mask=eval_mask,
            class_mask_true=class_mask_true,
            class_id=class_id,
        )
        record["boost"] = _summarize_effects(
            baseline_logp=baseline_logp,
            steered_logp=boost_logp,
            eval_mask=eval_mask,
            class_mask_true=class_mask_true,
            class_id=class_id,
        )
        class_results.append(record)
        print(
            f"Class {class_id:02d} | features={selected_fids} ({match_type}) | "
            f"ablate dlogp={record['ablate']['mean_delta_logprob_target_class']:.4f} | "
            f"boost dlogp={record['boost']['mean_delta_logprob_target_class']:.4f}"
        )

    output = {
        "config": {
            "gcn_ckpt": gcn_ckpt,
            "sae_ckpt": sae_ckpt,
            "feature_explanations": expl_path,
            "feature_stats": stats_path,
            "steer_layer": steer_layer,
            "min_purity": args.min_purity,
            "min_majority_count": args.min_majority_count,
            "boost_strength": args.boost_strength,
            "top_k_features": args.top_k_features,
            "all_nodes": args.all_nodes,
        },
        "results": class_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved steering results to: {out_path}")


if __name__ == "__main__":
    main()
