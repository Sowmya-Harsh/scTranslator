"""
temporal_proteotranslator.py
─────────────────────────────
Extension of scTranslator for temporal proteomics prediction.

WHAT THIS FILE DOES:
  Wraps the original scPerformerEncDec (performer_enc_dec.py — untouched)
  with three new capabilities:

  1. Time conditioning via FiLM (Feature-wise Linear Modulation):
       log(t+1) → TimeEmbedding → FiLMConditioner → scale+shift every RNA token
       This conditions EVERY gene token before Performer attention, not just
       a single prepended token.  FiLM is the standard approach in conditional
       generation models (DiT, ControlNet) because it modulates each feature
       directly rather than relying on attention to propagate a single token.

  2. Future-timepoint prediction:
       log(Δt+1) → DeltaTimeEmbedding → fused with time emb → FiLM
       RNA(t) + time(t) + horizon(Δt) → Protein(t + Δt)

  3. PTM ratio head:
       from predicted protein abundances, supervise the
       phospho/total ratio for matched pairs (p-p38/p38 etc.)

HOW IT EXTENDS THE ORIGINAL:
  Original scPerformerEncDec.forward(seq_in, seq_inID, seq_outID):
    → returns (encodings, protein_pred)

  TemporalProteoTranslator.forward(rna, gene_ids, prot_ids,
                                    log_t, log_delta_t):
    → FiLM-conditions RNA embeddings with fused time vector
    → runs original Performer encoder (unmodified)
    → returns (protein_pred, ratio_pred)

ORIGINAL CODE IS NOT MODIFIED:
  performer_enc_dec.py is imported as-is.
  No max_seq_len patching needed — FiLM does not prepend any token,
  so the encoder always receives exactly max_genes tokens.

Biological rationale (from QuRIE-seq paper, Rivello et al. 2021):
  - Phospho-proteome changes occur on MINUTE timescale (0-6 min)
  - Transcriptome changes occur on HOUR timescale (60-180 min)
  - RNA at time t contains information about FUTURE protein states
  - Δt embedding captures this prediction horizon explicitly
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from performer_enc_dec import scPerformerEncDec
from time_embedding import TimeEmbedding, DeltaTimeEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# FiLM Conditioner
# ─────────────────────────────────────────────────────────────────────────────

class FiLMConditioner(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) conditioning layer.

    Given a conditioning vector c (batch, dim), computes learned per-feature
    scale γ and shift β, then applies:

        output = γ * x + β

    where x is a sequence of token embeddings (batch, seq_len, dim).
    γ and β are broadcast over the sequence dimension so every gene token
    is modulated identically by the time signal.

    This is more powerful than prepending a single time token because:
      - Every position gets the time information directly (no attention hop)
      - The modulation is applied before attention, so attention patterns
        can adapt to the time-conditioned features
    """

    def __init__(self, dim: int):
        super().__init__()
        # Two-layer MLP maps conditioning vector → (gamma, beta) pair
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )

    def forward(
        self,
        cond: torch.Tensor,   # (batch, dim) — fused time+delta embedding
        x:    torch.Tensor,   # (batch, seq_len, dim) — RNA token embeddings
    ) -> torch.Tensor:
        """
        Returns x modulated by (gamma, beta) derived from cond.
        """
        params = self.net(cond)               # (batch, 2*dim)
        gamma, beta = params.chunk(2, dim=-1) # (batch, dim) each
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# PTM Ratio Head
# ─────────────────────────────────────────────────────────────────────────────

class PTMRatioHead(nn.Module):
    """
    Computes predicted activation ratios from protein predictions.

    For each matched phospho/total pair (e.g. p-p38/p38):
        ratio = log(phospho + ε) - log(total + ε)
              = log(phospho / total)

    Zero-parameter operation. Indices are resolved from ab_names at init
    time so the head is robust to changes in protein list ordering.

    All 11 pairs from QuRIE-seq qurie_phospho_pairs.csv.
    Only pairs whose names appear in ab_names are activated.
    """

    _PHOSPHO_PAIR_NAMES: List[Tuple[str, str]] = [
        ("p-Akt",    "Akt"),
        ("p-BLNK",   "BLNK"),
        ("p-Btk",    "Btk"),
        ("p-CD79a",  "CD79a"),
        ("p-Erk1/2", "Erk1/2"),
        ("p-JNK",    "JNK"),
        ("p-S6",     "S6"),
        ("p-SHP-1",  "SHP-1"),
        ("p-Syk",    "Syk"),
        ("p-p38",    "p38"),
        ("p-p65",    "p65"),
    ]

    def __init__(self, ab_names: List[str], eps: float = 0.1):
        super().__init__()
        self.eps = eps

        name_to_idx = {name: i for i, name in enumerate(ab_names)}
        valid_pairs = []
        for pname, tname in self._PHOSPHO_PAIR_NAMES:
            if pname in name_to_idx and tname in name_to_idx:
                valid_pairs.append(
                    (pname, name_to_idx[pname], tname, name_to_idx[tname])
                )
            else:
                print(f"  Warning: PTM pair {pname}/{tname} not in ab_names — skipping")

        if not valid_pairs:
            raise ValueError(
                "No valid PTM pairs found in ab_names. "
                "Check that phospho and total protein names are present."
            )

        self.phospho_idx = [p[1] for p in valid_pairs]
        self.total_idx   = [p[3] for p in valid_pairs]
        self.pair_names  = [f"{p[0]}/{p[2]}" for p in valid_pairs]
        self.n_pairs     = len(valid_pairs)

    def forward(self, protein_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein_pred: (batch, n_prot)
        Returns:
            ratio_pred:   (batch, n_pairs)  log-scale activation ratios
        """
        phospho = protein_pred[:, self.phospho_idx]
        total   = protein_pred[:, self.total_idx]
        ratio   = (torch.log(torch.clamp(phospho, min=0.0) + self.eps) -
                   torch.log(torch.clamp(total,   min=0.0) + self.eps))
        return ratio


# ─────────────────────────────────────────────────────────────────────────────
# TemporalProteoTranslator
# ─────────────────────────────────────────────────────────────────────────────

class TemporalProteoTranslator(nn.Module):
    """
    Temporal extension of scTranslator for proteomics prediction.

    Core change vs. original scTranslator: FiLM conditioning injects
    the time signal into every RNA token before Performer attention.

    Extends scPerformerEncDec with:
      1. TimeEmbedding + DeltaTimeEmbedding — produce (batch, dim) vectors
      2. FiLMConditioner — scales and shifts every RNA token by time signal
      3. PTMRatioHead — phospho/total ratio supervision (zero parameters)

    The original scPerformerEncDec is not modified.  No max_seq_len
    patching is required because FiLM does not prepend any extra token.

    Parameters
    ----------
    dim : int
        Hidden dimension (must match pretrained checkpoint, default 128)
    translator_depth : int
        MLP translator depth (default 2)
    initial_dropout : float
        Dropout in MLPTranslator (default 0.2)
    enc_max_seq_len : int
        Exactly max_genes — translator input size, matches pretrained.
    enc_depth, enc_heads : int
        Encoder Performer depth and heads
    dec_max_seq_len : int
        Number of proteins to predict (80 for QuRIE-seq)
    dec_depth, dec_heads : int
        Decoder Performer depth and heads
    ab_names : list of str
        Ordered protein/antibody names matching dec_max_seq_len.
        Used by PTMRatioHead to resolve phospho/total pair indices.
    use_delta_t : bool
        Whether to condition on prediction horizon Δt (default True)
    """

    def __init__(
        self,
        dim: int = 128,
        translator_depth: int = 2,
        initial_dropout: float = 0.2,
        enc_max_seq_len: int = 9000,
        enc_depth: int = 2,
        enc_heads: int = 8,
        dec_max_seq_len: int = 80,
        dec_depth: int = 2,
        dec_heads: int = 8,
        ab_names: Optional[List[str]] = None,
        use_delta_t: bool = True,
        **kwargs
    ):
        super().__init__()

        self.dim         = dim
        self.use_delta_t = use_delta_t

        # ── Core scTranslator (original, unmodified) ──────────────────────
        # enc_max_seq_len = max_genes exactly. No +1 needed because FiLM
        # does not prepend any token to the encoder input.
        self.core = scPerformerEncDec(
            dim              = dim,
            translator_depth = translator_depth,
            initial_dropout  = initial_dropout,
            enc_max_seq_len  = enc_max_seq_len,
            enc_depth        = enc_depth,
            enc_heads        = enc_heads,
            dec_max_seq_len  = dec_max_seq_len,
            dec_depth        = dec_depth,
            dec_heads        = dec_heads,
            dec_num_tokens   = 1,
        )

        # ── Time embedding modules → (batch, dim) vectors ─────────────────
        self.time_emb  = TimeEmbedding(dim)
        self.delta_emb = DeltaTimeEmbedding(dim)

        # ── Fuse norm: element-wise sum of t_emb and dt_emb ───────────────
        self.time_fuse_norm = nn.LayerNorm(dim)

        # ── FiLM conditioner: maps fused time vector → (gamma, beta) ──────
        # Applied to RNA embeddings before Performer attention so every
        # gene token is modulated by the time signal.
        self.film = FiLMConditioner(dim)

        # ── PTM ratio head (indices resolved from ab_names) ───────────────
        if ab_names is None:
            raise ValueError(
                "ab_names must be provided so PTMRatioHead can resolve "
                "phospho/total pair indices by name."
            )
        self.ratio_head = PTMRatioHead(ab_names)

        print(f"  [TemporalProteoTranslator] Initialized (FiLM conditioning)")
        print(f"    dim={dim}, enc_seq={enc_max_seq_len}, "
              f"dec_seq={dec_max_seq_len}")
        print(f"    use_delta_t={use_delta_t}")
        print(f"    PTM pairs: {self.ratio_head.pair_names}")
        n_params = sum(p.numel() for p in self.parameters())
        print(f"    Total params: {n_params:,}")

    def load_pretrained(
        self,
        checkpoint_path: str,
        map_location: str = "cpu"
    ) -> int:
        """
        Load weights from scTranslator pretrained checkpoint.
        Only loads layers whose names and shapes match exactly.
        New layers (time_emb, delta_emb, film, ratio_head) are randomly
        initialised.

        Returns:
            n_loaded: number of matching layers loaded
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict", ckpt)
        elif hasattr(ckpt, "state_dict"):
            state = ckpt.state_dict()
        else:
            state = ckpt

        our_state = self.state_dict()
        filtered  = {}
        for ck, cv in state.items():
            our_key = f"core.{ck}"
            if our_key in our_state and our_state[our_key].shape == cv.shape:
                filtered[our_key] = cv

        n_loaded = len(filtered)
        self.load_state_dict(filtered, strict=False)
        print(f"  Loaded {n_loaded} matching layers from {checkpoint_path}")
        return n_loaded

    def forward(
        self,
        rna:         torch.Tensor,                   # (batch, max_genes)
        gene_ids:    torch.Tensor,                   # (batch, max_genes)
        prot_ids:    torch.Tensor,                   # (batch, n_prot)
        log_t:       torch.Tensor,                   # (batch,)
        log_delta_t: Optional[torch.Tensor] = None,  # (batch,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            rna:         (batch, max_genes)   log1p RNA expression
            gene_ids:    (batch, max_genes)   scTranslator gene IDs
            prot_ids:    (batch, n_prot)      protein positional IDs
            log_t:       (batch,)             log1p(current time in minutes)
            log_delta_t: (batch,) or None     log1p(Δt in minutes);
                                              defaults to 0 if None

        Returns:
            protein_pred: (batch, n_prot)
            ratio_pred:   (batch, n_pairs)
        """
        # ── Compute fused time conditioning vector ────────────────────────
        t_emb = self.time_emb(log_t)              # (batch, dim)

        if self.use_delta_t:
            if log_delta_t is None:
                log_delta_t = torch.zeros_like(log_t)
            dt_emb = self.delta_emb(log_delta_t)  # (batch, dim)
            fused  = self.time_fuse_norm(t_emb + dt_emb)  # (batch, dim)
        else:
            fused = t_emb                          # (batch, dim)

        # ── FiLM-conditioned encoder + translator + decoder ───────────────
        protein_pred = self._forward_with_film(rna, gene_ids, prot_ids, fused)

        # ── PTM ratios ────────────────────────────────────────────────────
        ratio_pred = self.ratio_head(protein_pred)

        return protein_pred, ratio_pred

    def _forward_with_film(
        self,
        rna:        torch.Tensor,   # (batch, max_genes)
        gene_ids:   torch.Tensor,   # (batch, max_genes)
        prot_ids:   torch.Tensor,   # (batch, n_prot)
        fused_cond: torch.Tensor,   # (batch, dim) — fused time+delta vector
    ) -> torch.Tensor:
        """
        Runs the encoder with FiLM-conditioned RNA embeddings.

        Steps:
          1. Embed RNA scalars: Linear(1, dim)
          2. Add gene positional embeddings (indexed by gene ID)
          3. FiLM: apply gamma * x + beta from fused_cond
          4. Dropout + Performer attention
          5. Translate: MLP (max_genes → n_prot) over the dim axis
          6. Decode: Performer → scalar abundances
        """
        enc = self.core.enc

        # Step 1–2: RNA embed + positional
        x = rna
        if len(x.shape) < 3:
            x = x.unsqueeze(2)
        x = enc.to_vector(x)           # (batch, max_genes, dim)
        x += enc.pos_emb(gene_ids)     # (batch, max_genes, dim)

        # Step 3: FiLM — every gene token modulated by time signal
        x = self.film(fused_cond, x)   # (batch, max_genes, dim)

        # Step 4: dropout + Performer attention
        x = enc.dropout(x)
        layer_pos_emb = enc.layer_pos_emb(x)
        encodings = enc.performer(x, pos_emb=layer_pos_emb)
        # encodings: (batch, max_genes, dim) — no slicing needed (no prepended token)

        # Step 5: translate RNA → protein space
        seq_out = self.core.translator(
            encodings.transpose(1, 2).contiguous()
        ).transpose(1, 2).contiguous()  # (batch, n_prot, dim)

        # Step 6: decode to scalar abundances
        # scPerformerLM uses torch.squeeze() which removes ALL size-1 dims.
        # When batch_size=1 the output collapses to (n_prot,) instead of
        # (1, n_prot). Force back to (batch, n_prot) with view().
        batch = rna.shape[0]
        protein_pred = self.core.dec(seq_out, prot_ids)
        protein_pred = protein_pred.view(batch, -1)   # always (batch, n_prot)
        return protein_pred

    def get_trainable_param_groups(
        self,
        frozen_encoder: bool = True,
        lr_enc: float = 5e-7,
        lr_rest: float = 5e-6,
    ) -> list:
        """
        Returns Phase 1 parameter groups for AdamW.

        Phase 1 (frozen_encoder=True): encoder frozen, one param group.
        Phase 2 transition uses optimizer.add_param_group() in the training
        script to preserve translator/decoder momentum.
        """
        new_modules = [
            self.time_emb,
            self.delta_emb,
            self.film,            # FiLM conditioner
            self.ratio_head,
            self.time_fuse_norm,
        ]
        old_modules_no_enc = [
            self.core.translator,
            self.core.dec,
        ]

        new_params = []
        for m in new_modules:
            new_params.extend(list(m.parameters()))

        old_params = []
        for m in old_modules_no_enc:
            old_params.extend(list(m.parameters()))

        if frozen_encoder:
            for p in self.core.enc.parameters():
                p.requires_grad = False
            print(f"  Encoder frozen (Phase 1)")
            return [{"params": new_params + old_params, "lr": lr_rest}]
        else:
            for p in self.core.enc.parameters():
                p.requires_grad = True
            print(f"  Encoder unfrozen (lr={lr_enc:.1e})")
            enc_params = list(self.core.enc.parameters())
            return [
                {"params": enc_params,              "lr": lr_enc},
                {"params": new_params + old_params, "lr": lr_rest},
            ]
