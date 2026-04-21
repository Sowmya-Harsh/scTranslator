"""
qurie_dataset.py
─────────────────
PyTorch Dataset for QuRIE-seq single-cell data.

Two modes:
  SameTimepointDataset  — classic RNA(t) → Protein(t)  [Δt = 0]
  TemporalPairDataset   — RNA(t) → Protein(t+Δt)       [Δt > 0]

The training script uses BOTH datasets together via ConcatDataset,
which doubles the effective training signal.

Biological note (from QuRIE-seq paper):
  Phospho-proteome responds in MINUTES (0-6 min)
  Transcriptome responds in HOURS (60-180 min)
  → RNA at t=6 min already encodes what will happen at t=60 min
  → Temporal pairs capture this lag explicitly

Split strategy:
  Stratified 70/10/20 (train/val/test) within each timepoint and condition.
  Val split is used for early stopping — test split is held out for final eval.
  TemporalPairDataset is constructed from train indices only to prevent
  any cross-split data leakage.
  Ibrutinib condition (aIg+Ibru) kept separate for drug effect analysis.
"""

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Gene ID mapping
# ─────────────────────────────────────────────────────────────────────────────

def map_gene_ids(gene_names: List[str]) -> List[int]:
    """Map HUGO gene symbols → scTranslator internal IDs."""
    with open("code/model/ID_dic/hgs_to_EntrezID.pkl", "rb") as f:
        hgs_to_entrez = pickle.load(f)
    with open("code/model/ID_dic/EntrezID_to_myID.pkl", "rb") as f:
        entrez_to_myid = pickle.load(f)
    mapped = []
    for gene in gene_names:
        entrez = hgs_to_entrez.get(gene, None)
        if entrez is None:
            mapped.append(0)
            continue
        myid = entrez_to_myid.get(entrez, 0)
        try:    myid = int(myid)
        except: myid = 0
        mapped.append(myid)
    return mapped


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_qurie_processed(
    data_dir:      str,
    max_genes:     int   = 9000,
    seed:          int   = 42,
    val_fraction:  float = 0.1,
    test_fraction: float = 0.2,
) -> dict:
    """
    Load and preprocess QuRIE-seq arrays.

    Split strategy: stratified 70/10/20 (train/val/test) within each timepoint.
    val_idx is used for early stopping; test_idx is held out for final reporting.
    TemporalPairDataset only ever uses train_idx — no cross-split leakage.

    Returns a dict with keys:
        RNA, Protein, LogTime, meta, gene_ids, ab_names,
        train_idx, val_idx, test_idx, prot_ids
    """
    RNA     = np.load(f"{data_dir}/qurie_RNA.npy")
    Protein = np.load(f"{data_dir}/qurie_Protein.npy")
    LogTime = np.load(f"{data_dir}/qurie_logtime.npy")
    meta    = pd.read_csv(f"{data_dir}/qurie_meta.csv")

    with open(f"{data_dir}/qurie_gene_names.txt") as f:
        gene_names = [l.strip() for l in f.readlines()]
    with open(f"{data_dir}/qurie_Ab_names.txt") as f:
        ab_names = [l.strip() for l in f.readlines()]

    # ── Gene ID mapping ───────────────────────────────────────────────────
    gene_ids = map_gene_ids(gene_names)
    valid    = np.array([gid > 0 for gid in gene_ids])
    RNA      = RNA[:, valid]
    gene_ids = [gid for gid, v in zip(gene_ids, valid) if v]
    print(f"  Genes in vocabulary: {sum(valid):,} / {len(valid):,}")

    # ── Top max_genes by mean expression ─────────────────────────────────
    if RNA.shape[1] > max_genes:
        top_idx  = np.argsort(RNA.mean(axis=0))[-max_genes:]
        RNA      = RNA[:, top_idx]
        gene_ids = [gene_ids[i] for i in top_idx]

    # ── Pad to max_genes ─────────────────────────────────────────────────
    pad = max_genes - RNA.shape[1]
    if pad > 0:
        RNA      = np.pad(RNA, ((0, 0), (0, pad)))
        gene_ids = gene_ids + [0] * pad

    # ── Protein positional IDs ────────────────────────────────────────────
    # Ab features don't have NCBI IDs; use fixed range above gene ID space
    n_prot   = Protein.shape[1]
    prot_ids = list(range(60000, 60000 + n_prot))

    # ── Stratified train/val/test split within each timepoint (aIg only) ─
    # train ≈ 70%, val ≈ 10%, test ≈ 20% of cells per timepoint.
    # Splitting is done first so TemporalPairDataset never sees val/test cells.
    rng      = np.random.default_rng(seed)
    aig_mask = (meta["condition"] == "aIg").values
    aig_idx  = np.where(aig_mask)[0]
    aig_meta = meta.iloc[aig_idx].reset_index(drop=True)

    train_list, val_list, test_list = [], [], []
    for tp in sorted(aig_meta["time_min"].unique()):
        tp_local  = np.where(aig_meta["time_min"] == tp)[0]
        tp_global = aig_idx[tp_local]
        n         = len(tp_global)

        n_test = max(1, int(n * test_fraction))
        n_val  = max(1, int(n * val_fraction))
        # Ensure at least 1 train cell remains
        n_train = max(1, n - n_test - n_val)
        n_val   = max(1, n - n_test - n_train)

        shuffled = rng.permutation(n)
        test_local  = shuffled[:n_test]
        val_local   = shuffled[n_test:n_test + n_val]
        train_local = shuffled[n_test + n_val:]

        test_list.append(tp_global[test_local])
        val_list.append(tp_global[val_local])
        train_list.append(tp_global[train_local])

    train_idx = np.concatenate(train_list)
    val_idx   = np.concatenate(val_list)
    test_idx  = np.concatenate(test_list)

    total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"  Split — train: {len(train_idx):,}  val: {len(val_idx):,}  "
          f"test: {len(test_idx):,}  (total aIg: {total:,})")

    return {
        "RNA":       RNA,
        "Protein":   Protein,
        "LogTime":   LogTime,
        "meta":      meta,
        "gene_ids":  gene_ids,
        "ab_names":  ab_names,
        "prot_ids":  prot_ids,
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "test_idx":  test_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: Same-timepoint  RNA(t) → Protein(t)   [Δt = 0]
# ─────────────────────────────────────────────────────────────────────────────

class SameTimepointDataset(Dataset):
    """
    Classic same-timepoint prediction:
        RNA(t) + time(t) + Δt=0 → Protein(t)

    Each item:
        rna:         (max_genes,)   log1p RNA expression
        gene_ids:    (max_genes,)   scTranslator gene IDs
        log_t:       scalar         log1p(current time in minutes)
        log_delta_t: scalar         0.0  (same timepoint)
        protein:     (n_prot,)      log1p protein abundances
        prot_ids:    (n_prot,)      protein positional IDs
    """

    def __init__(self, data: dict, split: str = "train"):
        if split == "train":
            idx = data["train_idx"]
        elif split == "val":
            idx = data["val_idx"]
        elif split == "test":
            idx = data["test_idx"]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        self.RNA      = torch.tensor(data["RNA"][idx],     dtype=torch.float32)
        self.Protein  = torch.tensor(data["Protein"][idx], dtype=torch.float32)
        self.LogTime  = torch.tensor(data["LogTime"][idx], dtype=torch.float32)
        self.gene_ids = torch.tensor(data["gene_ids"],     dtype=torch.long)
        self.prot_ids = torch.tensor(data["prot_ids"],     dtype=torch.long)
        self.meta     = data["meta"].iloc[idx].reset_index(drop=True)
        self.ab_names = data["ab_names"]

        tp_counts = self.meta.groupby("time_min").size().to_dict()
        print(f"  [SameTP/{split}] {len(idx):,} cells | "
              f"timepoints: { {k:v for k,v in sorted(tp_counts.items())} }")

    def __len__(self):
        return len(self.RNA)

    def __getitem__(self, idx):
        return (
            self.RNA[idx],
            self.gene_ids,
            self.LogTime[idx],
            torch.tensor(0.0),       # Δt = 0 for same-timepoint
            self.Protein[idx],
            self.prot_ids,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: Temporal pairs  RNA(t) → Protein(t+Δt)  [Δt > 0]
# ─────────────────────────────────────────────────────────────────────────────

class TemporalPairDataset(Dataset):
    """
    Future-timepoint prediction:
        RNA(t) + time(t) + Δt → Protein(t + Δt)

    Captures the biological lag between RNA and protein:
        - RNA at t=6 min already encodes future protein at t=60 min
        - Transcription factor activity drives later proteome changes

    Construction:
        For each ordered pair of timepoints (t_src, t_tgt) where t_tgt > t_src:
            Randomly pair cells from t_src (RNA source)
            with cells from t_tgt (Protein target)

        This is a population-level prediction — cell_i and cell_j are
        DIFFERENT CELLS from the same isogenic BJAB population.
        Valid because all cells receive identical BCR stimulation.

    IMPORTANT: Only uses indices from the requested split so that training
    temporal pairs never include val or test cells as protein targets.

    Forward pairs used (aIg only):
        0  → 2, 4, 6, 60, 180 min
        2  → 4, 6, 60, 180 min
        4  → 6, 60, 180 min
        6  → 60, 180 min
        60 → 180 min
    """

    def __init__(
        self,
        data:      dict,
        split:     str = "train",
        seed:      int = 42,
        max_pairs: int = 5000,   # cap to avoid imbalance vs same-tp
    ):
        rng = np.random.default_rng(seed + 42)

        if split == "train":
            idx = data["train_idx"]
        elif split == "val":
            idx = data["val_idx"]
        elif split == "test":
            idx = data["test_idx"]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        meta_s  = data["meta"].iloc[idx].reset_index(drop=True)
        RNA_s   = data["RNA"][idx]
        Prot_s  = data["Protein"][idx]

        timepoints = sorted(meta_s["time_min"].unique())

        # Build all (source, target) pairs within this split only
        rna_list, prot_list, log_t_list, log_dt_list = [], [], [], []

        for i, t_src in enumerate(timepoints):
            for t_tgt in timepoints[i+1:]:
                src_mask = (meta_s["time_min"] == t_src).values
                tgt_mask = (meta_s["time_min"] == t_tgt).values

                src_idx = np.where(src_mask)[0]
                tgt_idx = np.where(tgt_mask)[0]

                if len(src_idx) == 0 or len(tgt_idx) == 0:
                    continue

                n = min(len(src_idx), len(tgt_idx))
                chosen_src = rng.choice(src_idx, size=n, replace=False)
                chosen_tgt = rng.choice(tgt_idx, size=n, replace=False)

                log_t  = float(np.log1p(t_src))
                log_dt = float(np.log1p(t_tgt - t_src))

                rna_list.append(RNA_s[chosen_src])
                prot_list.append(Prot_s[chosen_tgt])
                log_t_list.extend([log_t]  * n)
                log_dt_list.extend([log_dt] * n)

        if not rna_list:
            raise ValueError("No temporal pairs could be constructed.")

        RNA_all   = np.vstack(rna_list)
        Prot_all  = np.vstack(prot_list)
        LogT_all  = np.array(log_t_list,  dtype=np.float32)
        LogDT_all = np.array(log_dt_list, dtype=np.float32)

        # Cap at max_pairs to avoid imbalance
        if len(RNA_all) > max_pairs:
            chosen    = rng.choice(len(RNA_all), size=max_pairs, replace=False)
            RNA_all   = RNA_all[chosen]
            Prot_all  = Prot_all[chosen]
            LogT_all  = LogT_all[chosen]
            LogDT_all = LogDT_all[chosen]

        self.RNA      = torch.tensor(RNA_all,   dtype=torch.float32)
        self.Protein  = torch.tensor(Prot_all,  dtype=torch.float32)
        self.LogTime  = torch.tensor(LogT_all,  dtype=torch.float32)
        self.LogDelta = torch.tensor(LogDT_all, dtype=torch.float32)
        self.gene_ids = torch.tensor(data["gene_ids"], dtype=torch.long)
        self.prot_ids = torch.tensor(data["prot_ids"], dtype=torch.long)
        self.ab_names = data["ab_names"]

        print(f"  [TemporalPairs/{split}] {len(self.RNA):,} pairs")

    def __len__(self):
        return len(self.RNA)

    def __getitem__(self, idx):
        return (
            self.RNA[idx],
            self.gene_ids,
            self.LogTime[idx],
            self.LogDelta[idx],     # Δt > 0
            self.Protein[idx],
            self.prot_ids,
        )
