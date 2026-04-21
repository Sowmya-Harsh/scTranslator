"""
train_qurie_temporal.py
────────────────────────
Training script for TemporalProteoTranslator on QuRIE-seq data.

Architecture:
  TemporalProteoTranslator (temporal_proteotranslator.py)
    └── scPerformerEncDec (performer_enc_dec.py — ORIGINAL, untouched)
    └── TimeEmbedding + DeltaTimeEmbedding → (batch, dim) vectors
    └── FiLMConditioner — scale+shift every RNA token before attention
    └── PTMRatioHead — zero-parameter log phospho/total ratio supervision

Training data:
  SameTimepointDataset   RNA(t) → Protein(t)       [Δt=0]
  TemporalPairDataset    RNA(t) → Protein(t+Δt)    [Δt>0]
  Combined via ConcatDataset + WeightedRandomSampler
  (temporal pairs upsampled to temporal_weight fraction of batches)

Loss functions:
  L_total = L_protein + λ_ptm × L_ptm + λ_ratio × L_ratio

  L_protein = 1 - mean per-protein Pearson R  (directly optimises the metric)
  L_ptm     = 1 - mean per-phospho-protein Pearson R  (upweighted)
  L_ratio   = MSE over 4 activation ratios  (small n_pairs, MSE more stable)

Training phases:
  Phase 1 (epochs 1 → freeze_epochs):
    Encoder FROZEN. Translator, decoder, FiLM, time modules trained.
  Phase 2 (freeze_epochs → end):
    Encoder fine-tuned at 10× lower LR via optimizer.add_param_group()
    — preserves translator/decoder momentum across the transition.

Early stopping:
  Patience-based on VAL Pearson R (test set held out for final reporting).

Usage:
  python train_qurie_temporal.py \\
      --data_dir /path/to/QuRIE_processed \\
      --pretrain_checkpoint checkpoint/scTranslator_2M.pt \\
      --output_dir checkpoint/qurie_temporal
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

sys.path.append('code/model')

from temporal_proteotranslator import TemporalProteoTranslator
from qurie_dataset import load_qurie_processed, SameTimepointDataset, \
                          TemporalPairDataset


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def pearson_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    eps:  float = 1e-8,
) -> torch.Tensor:
    """
    1 - mean per-feature Pearson R, computed over the batch dimension.

    Directly optimises the evaluation metric rather than MSE.  Unlike MSE,
    this is scale-invariant: the model is rewarded for getting the ranking
    of samples right rather than hitting exact values.

    pred, true: (batch, n_features)
    Returns:    scalar in [0, 2]  (0 = perfect, 1 = no correlation, 2 = anti)
    """
    pred_c = pred - pred.mean(dim=0, keepdim=True)   # (batch, n_features)
    true_c = true - true.mean(dim=0, keepdim=True)
    num    = (pred_c * true_c).sum(dim=0)             # (n_features,)
    denom  = pred_c.norm(dim=0) * true_c.norm(dim=0) + eps
    return (1.0 - (num / denom)).mean()


_mse = nn.MSELoss()   # unused — kept as fallback


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:        TemporalProteoTranslator,
    loader:       DataLoader,
    device:       torch.device,
    ab_names:     list,
    lambda_ptm:   float,
    lambda_ratio: float,
    phospho_idx:  list,
    print_detail: bool = False,
    loader_label: str  = "eval",
) -> tuple:
    """
    Evaluate model on a DataLoader.

    Returns:
        (total_loss, mean_protein_R, mean_ratio_R)

    When print_detail=True, prints per-timepoint, per-horizon (if delta
    values vary), top/bottom proteins, and activation ratio breakdown.
    """
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    all_ratio_pred, all_ratio_true = [], []
    all_time, all_delta = [], []

    ratio_head = model.ratio_head

    with torch.no_grad():
        for rna, gene_ids, log_t, log_dt, prot, prot_ids in loader:
            rna      = rna.to(device)
            gene_ids = gene_ids.to(device)
            log_t    = log_t.to(device)
            log_dt   = log_dt.to(device)
            prot     = prot.to(device)
            prot_ids = prot_ids.to(device)

            pred, ratio_pred = model(rna, gene_ids, prot_ids, log_t, log_dt)
            true_ratio       = ratio_head(prot)

            l_prot  = pearson_loss(pred, prot)
            l_ptm   = pearson_loss(pred[:, phospho_idx], prot[:, phospho_idx])
            l_ratio = pearson_loss(ratio_pred, true_ratio)
            loss    = l_prot + lambda_ptm * l_ptm + lambda_ratio * l_ratio

            total_loss += loss.item()
            all_pred.append(pred.cpu().numpy())
            all_true.append(prot.cpu().numpy())
            all_ratio_pred.append(ratio_pred.cpu().numpy())
            all_ratio_true.append(true_ratio.cpu().numpy())
            all_time.append(log_t.cpu().numpy())
            all_delta.append(log_dt.cpu().numpy())

    pred_all       = np.vstack(all_pred)
    true_all       = np.vstack(all_true)
    ratio_pred_all = np.vstack(all_ratio_pred)
    ratio_true_all = np.vstack(all_ratio_true)
    time_all       = np.concatenate(all_time)
    delta_all      = np.concatenate(all_delta)

    # ── Per-protein Pearson R ─────────────────────────────────────────────
    prot_r = []
    for j in range(pred_all.shape[1]):
        if true_all[:, j].std() > 1e-6:
            p = pred_all[:, j]
            t = true_all[:, j]
            if np.isfinite(p).all() and np.isfinite(t).all():
                r, _ = pearsonr(p, t)
                prot_r.append(r if not np.isnan(r) else 0.0)
            else:
                prot_r.append(0.0)
        else:
            prot_r.append(np.nan)
    prot_r  = np.array(prot_r)
    valid_r = prot_r[~np.isnan(prot_r)]
    mean_r  = float(np.mean(valid_r)) if len(valid_r) > 0 else 0.0

    # ── Per-ratio Pearson R ───────────────────────────────────────────────
    ratio_r = []
    for j in range(ratio_pred_all.shape[1]):
        if ratio_true_all[:, j].std() > 1e-6:
            p = ratio_pred_all[:, j]
            t = ratio_true_all[:, j]
            if np.isfinite(p).all() and np.isfinite(t).all():
                r, _ = pearsonr(p, t)
                ratio_r.append(r if not np.isnan(r) else 0.0)
            else:
                ratio_r.append(0.0)
        else:
            ratio_r.append(np.nan)
    mean_ratio_r = float(np.nanmean(ratio_r)) if ratio_r else 0.0

    if print_detail:
        W = 58
        print(f"\n  [{loader_label}] {'─'*W}")

        # ── Per-timepoint breakdown ───────────────────────────────────────
        print(f"  Per-timepoint Pearson R (protein):")
        for lt in sorted(np.unique(time_all)):
            mask = np.abs(time_all - lt) < 0.01
            if mask.sum() < 5:
                continue
            tp_r  = [pearsonr(pred_all[mask, j], true_all[mask, j])[0]
                     for j in range(pred_all.shape[1])
                     if true_all[mask, j].std() > 1e-6]
            tp_r  = [r for r in tp_r if not np.isnan(r)]
            avg   = np.mean(tp_r) if tp_r else 0.0
            t_min = int(round(np.expm1(lt)))
            bar   = "█" * max(0, int(avg * 40))
            print(f"    t={t_min:>4} min  ({mask.sum():>4} cells)  "
                  f"R={avg:+.4f}  {bar}")

        # ── Per-horizon breakdown (only when delta values vary) ───────────
        unique_dt = np.unique(np.round(delta_all, 3))
        if len(unique_dt) > 1:
            print(f"\n  Per-horizon Pearson R (protein):")
            for ldt in sorted(unique_dt):
                mask = np.abs(delta_all - ldt) < 0.01
                if mask.sum() < 5:
                    continue
                dt_min = int(round(np.expm1(ldt)))
                h_r = [pearsonr(pred_all[mask, j], true_all[mask, j])[0]
                       for j in range(pred_all.shape[1])
                       if true_all[mask, j].std() > 1e-6]
                h_r = [r for r in h_r if not np.isnan(r)]
                avg  = np.mean(h_r) if h_r else 0.0
                bar  = "█" * max(0, int(avg * 40))
                print(f"    Δt={dt_min:>4} min  ({mask.sum():>4} pairs)  "
                      f"R={avg:+.4f}  {bar}")

        # ── Top/bottom proteins ───────────────────────────────────────────
        valid_idx   = np.where(~np.isnan(prot_r))[0]
        sorted_by_r = valid_idx[np.argsort(prot_r[valid_idx])]
        print(f"\n  Top 10 best predicted proteins:")
        for rank, j in enumerate(sorted_by_r[-10:][::-1]):
            bar = "█" * max(0, int(prot_r[j] * 20))
            print(f"    {rank+1:>2}. {ab_names[j]:<22} R={prot_r[j]:+.4f} {bar}")
        print(f"\n  Bottom 10 worst predicted proteins:")
        for rank, j in enumerate(sorted_by_r[:10]):
            print(f"    {rank+1:>2}. {ab_names[j]:<22} R={prot_r[j]:+.4f}")

        # ── Activation ratios ─────────────────────────────────────────────
        print(f"\n  Activation ratio Pearson R:")
        print(f"  {'─'*W}")
        for j, (pname, r) in enumerate(zip(model.ratio_head.pair_names, ratio_r)):
            bar = "█" * max(0, int(r * 20)) if not np.isnan(r) else ""
            print(f"    {pname:<20}  R={r:+.4f}  {bar}")
        print(f"    Mean ratio R: {mean_ratio_r:+.4f}")

        # ── Summary ───────────────────────────────────────────────────────
        print(f"\n  {'─'*W}")
        print(f"  Protein:  mean R={mean_r:+.4f}  "
              f"median={np.median(valid_r):+.4f}  "
              f"max={np.max(valid_r):+.4f}")
        print(f"  % R > 0.1: {100*np.mean(valid_r > 0.1):.0f}%  "
              f"| % R > 0.2: {100*np.mean(valid_r > 0.2):.0f}%  "
              f"| % R > 0.3: {100*np.mean(valid_r > 0.3):.0f}%")
        print(f"  Ratios:   mean R={mean_ratio_r:+.4f}")
        print(f"  {'─'*W}\n")

    return total_loss / len(loader), mean_r, mean_ratio_r


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
        default="/home/users/sjanmahanthi/Data_Test/QuRIE_processed")
    parser.add_argument("--pretrain_checkpoint",
        default="checkpoint/scTranslator_2M.pt")
    parser.add_argument("--output_dir",       default="checkpoint/qurie_temporal")
    parser.add_argument("--epochs",           type=int,   default=150)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=5e-6)
    parser.add_argument("--max_genes",        type=int,   default=9000)
    parser.add_argument("--val_fraction",     type=float, default=0.1,
        help="Fraction of aIg cells per timepoint held out for validation")
    parser.add_argument("--lambda_ptm",       type=float, default=0.5,
        help="Weight for phospho-only Pearson loss")
    parser.add_argument("--lambda_ratio",     type=float, default=0.3,
        help="Weight for activation ratio Pearson loss")
    parser.add_argument("--temporal_weight",  type=float, default=0.3,
        help="Target fraction of batches drawn from temporal pairs "
             "(rest from same-timepoint). 0.3 = temporal pairs are 30%% "
             "of training signal regardless of dataset size imbalance.")
    parser.add_argument("--freeze_epochs",    type=int,   default=30,
        help="Epochs to keep encoder frozen (Phase 1)")
    parser.add_argument("--patience",         type=int,   default=10,
        help="Early stopping: number of non-improving val evaluations "
             "(each eval window is 5 epochs, so patience=10 → 50 epochs)")
    parser.add_argument("--accum_steps",      type=int,   default=4,
        help="Gradient accumulation steps "
             "(effective batch = batch_size × accum_steps)")
    parser.add_argument("--detail_every",     type=int,   default=25)
    parser.add_argument("--max_pairs",        type=int,   default=5000,
        help="Max temporal pairs constructed per split")
    parser.add_argument("--no_delta_t",       action="store_true",
        help="Disable delta-t prediction (ablation)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  TemporalProteoTranslator — QuRIE-seq")
    print(f"{'='*60}")
    print(f"  Device:              {device}")
    print(f"  LR:                  {args.lr}")
    print(f"  Eff. batch size:     {args.batch_size * args.accum_steps}")
    print(f"  λ_ptm:               {args.lambda_ptm}")
    print(f"  λ_ratio:             {args.lambda_ratio}")
    print(f"  temporal_weight:     {args.temporal_weight}")
    print(f"  use_delta_t:         {not args.no_delta_t}")
    print(f"  freeze_epochs:       {args.freeze_epochs}")
    print(f"  patience (evals):    {args.patience}")
    print(f"  val_fraction:        {args.val_fraction}")
    print(f"  Loss: Pearson (prot + ptm + ratio)")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\n  Loading data...")
    data = load_qurie_processed(
        args.data_dir,
        max_genes=args.max_genes,
        val_fraction=args.val_fraction,
    )

    ab_names = data["ab_names"]
    n_prot   = data["Protein"].shape[1]
    phospho_idx = [i for i, name in enumerate(ab_names) if name.startswith("p-")]
    print(f"\n  Phospho features: {len(phospho_idx)} / {n_prot}")

    # ── Build datasets ────────────────────────────────────────────────────
    train_same = SameTimepointDataset(data, split="train")
    train_temp = TemporalPairDataset(data, split="train",
                                     max_pairs=args.max_pairs)
    train_combined = ConcatDataset([train_same, train_temp])

    val_same  = SameTimepointDataset(data, split="val")
    test_same = SameTimepointDataset(data, split="test")

    # ── Weighted sampler: upsample temporal pairs to temporal_weight ──────
    # Without this, ~37k same-tp samples dwarf ~5k temporal pairs in every
    # batch, diluting the novel temporal supervision signal.
    n_same = len(train_same)
    n_temp = len(train_temp)
    tw     = args.temporal_weight
    w_same = (1.0 - tw) / n_same
    w_temp = tw / n_temp
    sample_weights = [w_same] * n_same + [w_temp] * n_temp
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_combined),
        replacement=True,
    )
    print(f"\n  Sampler: same-tp weight={w_same:.2e}  "
          f"temporal weight={w_temp:.2e}  "
          f"(target {tw*100:.0f}% temporal batches)")

    train_loader = DataLoader(train_combined, batch_size=args.batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_same,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_same, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Build model ───────────────────────────────────────────────────────
    print(f"\n  Building model...")
    model = TemporalProteoTranslator(
        dim              = 128,
        translator_depth = 2,
        initial_dropout  = 0.2,
        enc_max_seq_len  = args.max_genes,   # exactly max_genes; FiLM adds no extra token
        enc_depth        = 2,
        enc_heads        = 8,
        dec_max_seq_len  = n_prot,
        dec_depth        = 2,
        dec_heads        = 8,
        ab_names         = ab_names,
        use_delta_t      = not args.no_delta_t,
    )

    if os.path.exists(args.pretrain_checkpoint):
        model.load_pretrained(args.pretrain_checkpoint)
    else:
        print(f"  Warning: checkpoint not found — training from scratch")

    model = model.to(device)

    # ── Phase 1 optimizer (encoder frozen) ───────────────────────────────
    param_groups = model.get_trainable_param_groups(
        frozen_encoder=True,
        lr_enc=args.lr * 0.1,
        lr_rest=args.lr,
    )
    optimizer = AdamW(param_groups, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    best_val_r     = -1.0
    best_epoch     = 0
    patience_count = 0
    log_rows       = []
    tag = "no_delta" if args.no_delta_t else "with_delta"

    print(f"\n{'='*60}")
    print(f"  {'Epoch':>5}  {'Train L':>9}  {'Val L':>9}  "
          f"{'Val R':>8}  {'Ratio R':>8}  {'Phase':>10}")
    print(f"  {'─'*60}")

    for epoch in range(1, args.epochs + 1):

        # ── Phase 2 transition ────────────────────────────────────────────
        # add_param_group preserves translator/decoder momentum — no LR spike.
        if epoch == args.freeze_epochs + 1:
            for p in model.core.enc.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params":       list(model.core.enc.parameters()),
                "lr":           args.lr * 0.1,
                "weight_decay": 1e-2,
            })
            print(f"\n  Phase 2: encoder unfrozen "
                  f"(lr={args.lr*0.1:.1e}, added to existing optimizer)\n")

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (rna, gene_ids, log_t, log_dt, prot, prot_ids) in \
                enumerate(train_loader):

            rna      = rna.to(device)
            gene_ids = gene_ids.to(device)
            log_t    = log_t.to(device)
            log_dt   = log_dt.to(device)
            prot     = prot.to(device)
            prot_ids = prot_ids.to(device)

            pred, ratio_pred = model(rna, gene_ids, prot_ids, log_t, log_dt)
            true_ratio       = model.ratio_head(prot)

            # Pearson loss for proteins/PTMs; MSE for ratios (only 4 pairs)
            l_prot  = pearson_loss(pred, prot)
            l_ptm   = pearson_loss(pred[:, phospho_idx], prot[:, phospho_idx])
            l_ratio = pearson_loss(ratio_pred, true_ratio)
            loss    = (l_prot
                       + args.lambda_ptm   * l_ptm
                       + args.lambda_ratio * l_ratio) / args.accum_steps

            loss.backward()
            train_loss += loss.item() * args.accum_steps

            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(train_loader) % args.accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        train_loss /= len(train_loader)
        phase = "frozen" if epoch <= args.freeze_epochs else "full"

        # ── Evaluation ────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            print_detail = (epoch % args.detail_every == 0)

            # Val — early stopping metric
            val_loss, val_r, val_ratio_r = evaluate(
                model, val_loader, device, ab_names,
                args.lambda_ptm, args.lambda_ratio, phospho_idx,
                print_detail=print_detail, loader_label="val",
            )

            # Test — monitoring only, printed on detail epochs
            if print_detail:
                _, test_r, test_ratio_r = evaluate(
                    model, test_loader, device, ab_names,
                    args.lambda_ptm, args.lambda_ratio, phospho_idx,
                    print_detail=True, loader_label="test",
                )
            else:
                test_r, test_ratio_r = float("nan"), float("nan")

            note = f"[{phase}]"
            if val_r > best_val_r:
                best_val_r     = val_r
                best_epoch     = epoch
                patience_count = 0
                note          += " ✅"
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_pearson_r":    val_r,
                    "val_ratio_r":      val_ratio_r,
                    "args":             vars(args),
                }, f"{args.output_dir}/best_{tag}.pt")
            else:
                patience_count += 1

            print(f"  {epoch:>5}  {train_loss:>9.4f}  {val_loss:>9.4f}  "
                  f"{val_r:>+8.4f}  {val_ratio_r:>+8.4f}  {note}")

            log_rows.append({
                "epoch":       epoch,
                "train_loss":  train_loss,
                "val_loss":    val_loss,
                "val_r":       val_r,
                "val_ratio_r": val_ratio_r,
                "test_r":      test_r,
                "phase":       phase,
            })

            if patience_count >= args.patience and epoch > args.freeze_epochs:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val R: {best_val_r:+.4f} at epoch {best_epoch})")
                break

    # ── Final evaluation on held-out test set ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION — best model (epoch {best_epoch}, "
          f"val R={best_val_r:+.4f})")
    print(f"{'='*60}")
    ckpt = torch.load(f"{args.output_dir}/best_{tag}.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, final_test_r, final_ratio_r = evaluate(
        model, test_loader, device, ab_names,
        args.lambda_ptm, args.lambda_ratio, phospho_idx,
        print_detail=True, loader_label="test (final)",
    )

    # Per-horizon evaluation on test temporal pairs
    try:
        test_temp = TemporalPairDataset(data, split="test",
                                         max_pairs=args.max_pairs)
        test_temp_loader = DataLoader(test_temp, batch_size=args.batch_size,
                                      shuffle=False, num_workers=2,
                                      pin_memory=True)
        print(f"\n  Per-horizon evaluation on test temporal pairs:")
        evaluate(
            model, test_temp_loader, device, ab_names,
            args.lambda_ptm, args.lambda_ratio, phospho_idx,
            print_detail=True, loader_label="test temporal",
        )
    except Exception as e:
        print(f"  (Per-horizon eval skipped: {e})")

    pd.DataFrame(log_rows).to_csv(
        f"{args.output_dir}/log_{tag}.csv", index=False)
    print(f"\n  Best val Pearson R:   {best_val_r:+.4f}  (epoch {best_epoch})")
    print(f"  Final test Pearson R: {final_test_r:+.4f}")
    print(f"  Final ratio R:        {final_ratio_r:+.4f}")
    print(f"  Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
