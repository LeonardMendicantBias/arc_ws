#!/usr/bin/env python3
"""
finetune_mask_vae.py — Fine-tune DALL-E dVAE with MASK_TOKEN support.

The DALL-E dVAE encodes a 256×256 image to a 32×32 grid of 8192-way
soft one-hot codewords.  This script adds a learned MASK_TOKEN that
replaces a subset of those codeword positions before decoding, teaching
the decoder to reconstruct from partial latent maps (selective transmission).

MASK_TOKEN selection
    A calibration pass counts argmax-codeword frequency over training data.
    The single least-used codeword index is designated the MASK_TOKEN slot.
    Its position in the soft one-hot space is replaced by a trainable
    nn.Parameter so the decoder receives a distinct "missing" signal rather
    than a recycled rare codeword.

Freezing strategy (efficient fine-tuning)
    Frozen   encoder (all)
    Frozen   decoder.blocks.input + group_1 + group_2  (low-level features)
    Trained  decoder.blocks.group_3 + group_4 + output (high-level synthesis)
    Trained  mask_token   nn.Parameter [vocab_size]

Precision
    Encoder/Decoder Conv2d layers already cast to fp16 internally on CUDA.
    torch.cuda.amp GradScaler guards the backward pass against fp16 underflow.
    Inputs to encoder/decoder are kept float32 (their own type checks require it).

Usage
    # Install dependency (if not already):
    #   pip install git+https://github.com/openai/dall-e.git

    # Download pretrained weights:
    #   python finetune_mask_vae.py --download_weights

    # Evaluate baseline on test set (no fine-tuning, no masking):
    #   python finetune_mask_vae.py --eval

    # Evaluate baseline WITH 50% masking applied:
    #   python finetune_mask_vae.py --eval --mask_ratio 0.5

    # Calibrate codeword frequencies only:
    #   python finetune_mask_vae.py --calibrate_only

    # Fine-tune with 50% random spatial masking:
    #   python finetune_mask_vae.py --mask_ratio 0.5 --epochs 10

    # Resume fine-tuning from checkpoint:
    #   python finetune_mask_vae.py --mask_ratio 0.5 --resume checkpoints/best_model.pt
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile

# Raise on truncated files instead of silently returning partial pixel data.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ── dall-e ─────────────────────────────────────────────────────────────────
try:
    from dall_e import load_model, map_pixels, unmap_pixels
except ImportError:
    sys.exit(
        "dall_e not found.\n"
        "Install with:  pip install git+https://github.com/openai/dall-e.git"
    )

# ── constants ───────────────────────────────────────────────────────────────
VOCAB_SIZE        = 8192
IMAGE_SIZE        = 256          # encoder expects 256×256
LATENT_SIZE       = 32           # 256 / 8 = 32  (spatial resolution of z)
LOGIT_LAPLACE_EPS = 0.1          # same as dall_e.utils.logit_laplace_eps

ENC_URL      = "https://cdn.openai.com/dall-e/encoder.pkl"
DEC_URL      = "https://cdn.openai.com/dall-e/decoder.pkl"
DATASET_ROOT = Path("E:/OpenImageV7")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE,
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),   # [0, 1] float32
    ])


def _find_split_dir(root: Path, split: str) -> Path:
    """Locate a dataset split under several common naming conventions."""
    for name in [split, split.capitalize(), split.upper(),
                 split[:3], split[:3].capitalize()]:
        p = root / name
        if p.is_dir():
            return p
    # Fall back to root itself (flat structure)
    return root


class FlatImageDataset(Dataset):
    """Recursively collects all images beneath a directory.
    Works whether images are organised in class sub-folders or flat."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def __init__(self, root: Path, transform=None):
        self.paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No images found under {root}.\n"
                "Make sure E:/OpenImageV7 contains train/validation/test sub-folders "
                "and the dataset has finished downloading."
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            # Signal the collate function to drop this sample.
            return None
        if self.transform:
            img = self.transform(img)
        return img


def collate_skip_none(batch):
    """Drop None entries (corrupt/truncated files) from the batch.
    Returns None when every sample in the batch was bad — the training
    loop skips None batches entirely."""
    good = [x for x in batch if x is not None]
    if not good:
        return None
    return torch.stack(good)


def get_loader(split: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    root = _find_split_dir(DATASET_ROOT, split)
    ds = FlatImageDataset(root, transform=_build_transform())
    print(f"  {split}: {len(ds):,} images found under {root}")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_skip_none,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class MaskedDALLEVAE(nn.Module):
    """
    DALL-E dVAE extended with a learned MASK_TOKEN.

    During the forward pass a random boolean mask is generated at the latent
    spatial resolution (32×32).  Masked positions in the soft one-hot tensor
    are replaced by softmax(mask_token) before being fed to the decoder.

    Attributes
    ----------
    mask_token_id : int
        Codeword index designated as the MASK_TOKEN (set by calibration).
        Stored so checkpoints are self-contained.
    mask_token : nn.Parameter [vocab_size]
        Learnable substitution vector in logit space.
        Softmax'd before substitution to produce a valid distribution.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 mask_token_id: int = 0) -> None:
        super().__init__()
        self.encoder       = encoder
        self.decoder       = decoder
        self.mask_token_id = mask_token_id

        # Initialise close to a one-hot at the designated slot.
        init = torch.zeros(VOCAB_SIZE)
        init[mask_token_id] = 3.0        # softmax-peak ≈ 0.95 at this index
        self.mask_token = nn.Parameter(init)

    # ── encoding ─────────────────────────────────────────────────────────

    def encode_logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] float32 (map_pixels applied) → [B,V,h,w] logits."""
        return self.encoder(x.float())

    def logits_to_soft(self, logits: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
        """Soft one-hot from encoder logits, shape [B,V,h,w]."""
        return F.softmax(logits.float() / temperature, dim=1)

    # ── masking ──────────────────────────────────────────────────────────

    def _make_random_mask(self, B: int, h: int, w: int,
                          mask_ratio: float,
                          device: torch.device) -> torch.Tensor:
        """Returns bool tensor [B,h,w]; True = masked position."""
        n_mask = max(1, int(mask_ratio * h * w))
        mask = torch.zeros(B, h * w, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(h * w, device=device)[:n_mask]
            mask[b, idx] = True
        return mask.view(B, h, w)

    def apply_mask(self, soft: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """
        Replace masked positions with the learned MASK_TOKEN distribution.

        soft : [B,V,h,w]  soft one-hot from encoder
        mask : [B,h,w]    bool — True = mask this position
        """
        mt = self.mask_token_distribution()              # [V]
        mt = mt.view(1, VOCAB_SIZE, 1, 1)                # broadcast over B,h,w
        return torch.where(mask.unsqueeze(1), mt.expand_as(soft), soft)

    def mask_token_distribution(self) -> torch.Tensor:
        """Return softmax(mask_token) as a [V] float32 tensor.

        This is the same distribution `apply_mask` substitutes at masked
        positions; it is what downstream deployment (e.g. the TRT decoder
        engine running in server_cpp) should write into non-transmitted
        spatial columns of the decoder input tensor.
        """
        return F.softmax(self.mask_token.float(), dim=0)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                mask_ratio: float = 0.0):
        """
        Parameters
        ----------
        x          : [B,3,H,W] float32 — pixels after map_pixels()
        mask_ratio : fraction of latent positions to mask (0 = no masking)

        Returns
        -------
        recon_params : [B,6,H,W]  logit-Laplace parameters from decoder
        soft         : [B,V,h,w]  unmasked soft one-hot
        mask         : [B,h,w]    bool mask applied
        """
        logits = self.encode_logits(x)          # [B,V,h,w] — fp16 inside enc
        soft   = self.logits_to_soft(logits)    # float32 [B,V,h,w]

        B, _V, h, w = soft.shape
        mask = self._make_random_mask(B, h, w, mask_ratio, x.device)

        soft_in = self.apply_mask(soft, mask) if mask_ratio > 0 else soft
        recon   = self.decoder(soft_in.float())  # fp16 inside dec, float32 I/O

        return recon, soft, mask


def load_finetuned_model(
        checkpoint_path: str | Path,
        encoder_pkl: str | Path,
        decoder_pkl: str | Path,
        device: torch.device,
) -> "MaskedDALLEVAE":
    """Reconstruct a fine-tuned MaskedDALLEVAE from a saved checkpoint.

    The encoder/decoder are first instantiated from the pretrained .pkl files
    (the checkpoint stores parameter tensors, not module definitions); then
    the full state dict is loaded in. Returns the model in eval() mode on
    ``device`` with grads disabled — suitable for export / inference only.
    """
    encoder = load_model(str(encoder_pkl), device=device)
    decoder = load_model(str(decoder_pkl), device=device)

    ckpt = torch.load(str(checkpoint_path), map_location=device,
                      weights_only=False)
    mask_token_id = ckpt.get("mask_token_id", 0)
    model = MaskedDALLEVAE(encoder, decoder, mask_token_id=mask_token_id)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════

def logit_laplace_nll(recon_params: torch.Tensor,
                      target_mapped: torch.Tensor) -> torch.Tensor:
    """
    Logit-Laplace negative log-likelihood.

    recon_params  : [B,6,H,W]  (mu_logit × 3  ||  log_b × 3)
    target_mapped : [B,3,H,W]  pixels after map_pixels — in (eps, 1-eps)

    NLL = log(b) + |logit(t) - mu|/b + log(t) + log(1-t)
    """
    recon_params  = recon_params.float()
    target_mapped = target_mapped.float()

    # Guard against fp16 overflow in the decoder producing Inf/NaN.
    # clamp(-50, 50) keeps mu finite; logb clamped separately to [-20, 20]
    # so that exp(logb) stays in [~2e-9, ~5e8] — safe for float32 arithmetic.
    mu   = recon_params[:, :3].clamp(-50.0, 50.0)
    logb = recon_params[:, 3:].clamp(-20.0, 20.0)
    b    = logb.exp()                                # always finite, always > 0

    t = target_mapped.clamp(LOGIT_LAPLACE_EPS, 1.0 - LOGIT_LAPLACE_EPS)
    t_logit = torch.log(t) - torch.log1p(-t)

    nll = logb + (t_logit - mu).abs() / b + torch.log(t) + torch.log1p(-t)
    return nll.mean()


# ═══════════════════════════════════════════════════════════════════════════
# Calibration — find least-used codeword
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def calibrate_codeword_frequencies(
        encoder: nn.Module,
        loader: DataLoader,
        device: torch.device,
        max_batches: int = 500,
) -> torch.Tensor:
    """
    Returns a [vocab_size] LongTensor of argmax-codeword counts collected
    from up to `max_batches` training batches.
    """
    counts  = torch.zeros(VOCAB_SIZE, dtype=torch.long)
    encoder.eval()

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x      = map_pixels(batch.to(device))
        logits = encoder(x.float())              # [B,V,h,w]
        ids    = logits.argmax(dim=1).cpu().reshape(-1)
        counts.scatter_add_(0, ids, torch.ones_like(ids))

        if (i + 1) % 100 == 0:
            print(f"  calibration  {i + 1}/{max_batches} batches")

    encoder.train()
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# Freeze helpers
# ═══════════════════════════════════════════════════════════════════════════

def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(value)


def _get_submodule(root: nn.Module, dotted_name: str) -> nn.Module:
    m = root
    for part in dotted_name.split("."):
        m = getattr(m, part)
    return m


def apply_freeze_strategy(model: MaskedDALLEVAE) -> None:
    """
    Encoder         → fully frozen
    Decoder input + group_1 + group_2 → frozen
    Decoder group_3 + group_4 + output → trainable (requires_grad=True)
    mask_token      → trainable (already True as a fresh nn.Parameter)
    """
    # Encoder starts with requires_grad=False (pretrained default), keep it.
    _set_requires_grad(model.encoder, False)

    # Decoder — lower groups frozen, upper groups trainable.
    for name in ("blocks.input", "blocks.group_1", "blocks.group_2"):
        _set_requires_grad(_get_submodule(model.decoder, name), False)

    for name in ("blocks.group_3", "blocks.group_4", "blocks.output"):
        _set_requires_grad(_get_submodule(model.decoder, name), True)

    model.mask_token.requires_grad_(True)


def _count_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
        model: MaskedDALLEVAE,
        loader: DataLoader,
        device: torch.device,
        mask_ratio: float = 0.0,
        max_batches: int = 500,
        label: str = "",
) -> dict:
    model.eval()
    total_nll  = 0.0
    total_psnr = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if batch is None:
            continue
        x        = batch.to(device)          # [B,3,H,W] in [0,1]
        x_mapped = map_pixels(x)             # in (eps, 1-eps)

        recon_params, _soft, _mask = model(x_mapped, mask_ratio=mask_ratio)

        nll = logit_laplace_nll(recon_params, x_mapped)
        total_nll += nll.item()

        # Pixel-space reconstruction for PSNR: sigmoid(mu) ≈ mapped pixel,
        # then unmap back to [0,1].
        mu_mapped = torch.sigmoid(recon_params[:, :3].float())
        mu_pixel  = unmap_pixels(mu_mapped)
        mse       = F.mse_loss(mu_pixel, x.float(), reduction="mean").item()
        total_psnr += 10.0 * math.log10(1.0 / (mse + 1e-10))
        n += 1

    model.train()
    prefix = f"[{label}] " if label else ""
    return {
        f"{prefix}nll":              round(total_nll  / max(n, 1), 6),
        f"{prefix}psnr_db":          round(total_psnr / max(n, 1), 3),
        f"{prefix}mask_ratio":       mask_ratio,
        f"{prefix}batches_evaluated": n,
    }


def run_eval_suite(model: MaskedDALLEVAE,
                   loader: DataLoader,
                   device: torch.device,
                   mask_ratio: float,
                   max_batches: int = 500) -> dict:
    """Runs unmasked (baseline) and masked evaluation; prints and returns stats."""
    print("\n── Unmasked baseline ──────────────────────────────────────────")
    stats_base = evaluate(model, loader, device,
                          mask_ratio=0.0, max_batches=max_batches,
                          label="unmasked")
    for k, v in stats_base.items():
        print(f"  {k}: {v}")

    all_stats = dict(stats_base)

    if mask_ratio > 0:
        print(f"\n── Masked  (ratio={mask_ratio}) ─────────────────────────────────")
        stats_mask = evaluate(model, loader, device,
                              mask_ratio=mask_ratio, max_batches=max_batches,
                              label=f"masked_{mask_ratio:.2f}")
        for k, v in stats_mask.items():
            print(f"  {k}: {v}")

        # Delta PSNR
        psnr_base = stats_base["[unmasked] psnr_db"]
        psnr_mask = stats_mask[f"[masked_{mask_ratio:.2f}] psnr_db"]
        delta = psnr_base - psnr_mask
        print(f"\n  PSNR drop due to masking: {delta:.3f} dB")
        all_stats.update(stats_mask)
        all_stats["psnr_drop_db"] = round(delta, 3)

    return all_stats


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace,
          model: MaskedDALLEVAE,
          device: torch.device) -> None:

    train_loader = get_loader("train",      args.batch_size, args.num_workers)
    val_loader   = get_loader("validation", args.batch_size, args.num_workers)

    # Two parameter groups with distinct learning rates:
    #   mask_token   — brand-new parameter, learns at the full LR
    #   decoder upper layers — pretrained weights, need a much gentler LR
    #     to avoid corrupting the features learned during pretraining.
    decoder_lr = args.lr * args.decoder_lr_scale
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.mask_token],
             "lr": args.lr,
             "name": "mask_token"},
            {"params": [p for p in model.decoder.parameters()
                        if p.requires_grad],
             "lr": decoder_lr,
             "name": "decoder_upper"},
        ],
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )

    # GradScaler — use the device-type-aware torch.amp API
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = min(args.warmup_steps, total_steps // 10)

    def _lr_lambda(step: int) -> float:
        """Linear warmup then cosine decay, applied uniformly to all groups."""
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.01 + 0.99 * cosine   # decays to 1 % of base LR

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    print(
        f"  LR  mask_token    : {args.lr:.1e}\n"
        f"  LR  decoder upper : {decoder_lr:.1e}  "
        f"(scale={args.decoder_lr_scale})\n"
        f"  Warmup steps      : {warmup_steps}\n"
        f"  Total steps       : {total_steps}\n"
    )

    ckpt_dir  = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path  = ckpt_dir / "train_log.jsonl"

    best_val_nll  = float("inf")
    start_epoch   = 0
    global_step   = 0

    # Optionally restore optimizer state when resuming
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = start_epoch * len(train_loader)
        print(f"Resuming from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_nll = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if batch is None:           # entire batch was corrupt — skip
                continue
            x        = batch.to(device)
            x_mapped = map_pixels(x)

            optimizer.zero_grad(set_to_none=True)

            # fp16-aware forward (encoder/decoder manage own dtypes internally)
            with autocast("cuda", enabled=(device.type == "cuda")):
                recon_params, _soft, _mask = model(x_mapped,
                                                   mask_ratio=args.mask_ratio)
                loss = logit_laplace_nll(recon_params, x_mapped)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only advance the LR schedule when optimizer.step() actually ran.
            # If Inf/NaN was detected, GradScaler skips optimizer.step() and
            # halves the scale — scheduler.step() before optimizer.step() would
            # produce the PyTorch 1.1 ordering warning and skip the first LR value.
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            loss_val = loss.item()
            epoch_nll += loss_val if math.isfinite(loss_val) else 0.0
            global_step += 1

            if step % args.log_every == 0:
                lrs    = scheduler.get_last_lr()
                lr_mt  = lrs[0]           # mask_token group
                lr_dec = lrs[1]           # decoder group
                print(f"  epoch {epoch + 1}/{args.epochs}  "
                      f"step {step:5d}/{len(train_loader)}  "
                      f"nll={loss_val:.4f}  "
                      f"lr_mt={lr_mt:.2e}  lr_dec={lr_dec:.2e}")

        # ── end of epoch ──────────────────────────────────────────────────
        mean_train_nll = epoch_nll / len(train_loader)
        val_stats      = evaluate(model, val_loader, device,
                                  mask_ratio=args.mask_ratio, max_batches=100)
        val_nll  = val_stats["nll"]
        val_psnr = val_stats["psnr_db"]
        elapsed  = time.time() - t0

        print(
            f"\n[epoch {epoch + 1}]  "
            f"train_nll={mean_train_nll:.4f}  "
            f"val_nll={val_nll:.4f}  "
            f"val_psnr={val_psnr:.2f} dB  "
            f"time={elapsed:.0f}s\n"
        )

        log_row = {
            "epoch": epoch + 1,
            "train_nll": round(mean_train_nll, 6),
            "val_nll":   round(val_nll,        6),
            "val_psnr":  round(val_psnr,        3),
            "mask_ratio": args.mask_ratio,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_row) + "\n")

        # Save best checkpoint
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_nll":              best_val_nll,
                "mask_token_id":        model.mask_token_id,
                "args":                 vars(args),
            }, ckpt_path)
            print(f"  saved best checkpoint → {ckpt_path}")

    print(f"\nTraining complete.  Best val NLL: {best_val_nll:.6f}")
    print(f"Log: {log_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Weight download utility
# ═══════════════════════════════════════════════════════════════════════════

def download_weights(enc_path: str, dec_path: str) -> None:
    import urllib.request

    for url, path in [(ENC_URL, enc_path), (DEC_URL, dec_path)]:
        if Path(path).exists():
            print(f"  {path} already exists, skipping.")
            continue
        print(f"  Downloading {url} → {path} …")
        urllib.request.urlretrieve(url, path)
        print(f"  Done.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune DALL-E dVAE with a MASK_TOKEN for selective "
                    "codeword transmission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Weights
    p.add_argument("--encoder_path", default="encoder.pkl",
                   help="Path (or URL) to pretrained DALL-E encoder weights")
    p.add_argument("--decoder_path", default="decoder.pkl",
                   help="Path (or URL) to pretrained DALL-E decoder weights")
    p.add_argument("--download_weights", action="store_true",
                   help="Download official pretrained weights then exit")

    # Output
    p.add_argument("--output_dir", default="checkpoints",
                   help="Directory for checkpoints, logs and eval results")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint to resume fine-tuning from")

    # Masking
    p.add_argument("--mask_ratio", type=float, default=0.5,
                   help="Fraction of 32×32 latent positions to mask per image")

    # Training hyper-parameters
    p.add_argument("--epochs",            type=int,   default=5)
    p.add_argument("--batch_size",        type=int,   default=16)
    p.add_argument("--lr",                type=float, default=3e-5,
                   help="Base LR for mask_token (new parameter)")
    p.add_argument("--decoder_lr_scale",  type=float, default=0.1,
                   help="Multiplier on --lr for pretrained decoder layers "
                        "(default 0.1 → decoder LR = lr × 0.1)")
    p.add_argument("--warmup_steps",      type=int,   default=500,
                   help="Linear LR warm-up steps before cosine decay kicks in")
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--log_every",         type=int,   default=50,
                   help="Print training stats every N steps")

    # Calibration
    p.add_argument("--calib_batches", type=int, default=500,
                   help="Training batches to use for codeword frequency calibration")
    p.add_argument("--calibrate_only", action="store_true",
                   help="Run calibration, print frequency statistics, then exit")

    # Evaluation
    p.add_argument("--eval", action="store_true",
                   help="Evaluate on test set and exit (no fine-tuning)")
    p.add_argument("--eval_batches", type=int, default=500,
                   help="Max batches used during evaluation")

    # Device
    p.add_argument("--device", default=None,
                   help="Force a specific device, e.g. cuda:0 or cpu")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")

    # ── optional weight download ───────────────────────────────────────────
    if args.download_weights:
        download_weights(args.encoder_path, args.decoder_path)
        return

    # ── load pretrained models ─────────────────────────────────────────────
    print(f"Loading encoder from {args.encoder_path} …")
    encoder = load_model(args.encoder_path, device=device)
    print(f"Loading decoder from {args.decoder_path} …")
    decoder = load_model(args.decoder_path, device=device)

    # ── calibrate codeword frequencies ────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    calib_cache = Path(args.output_dir) / "codeword_counts.pt"

    if calib_cache.exists() and not args.calibrate_only:
        print(f"Loading cached codeword counts from {calib_cache}")
        counts = torch.load(calib_cache, map_location="cpu")
    else:
        print("Calibrating codeword frequencies …")
        calib_loader = get_loader("train", args.batch_size, args.num_workers)
        counts = calibrate_codeword_frequencies(
            encoder, calib_loader, device, max_batches=args.calib_batches
        )
        torch.save(counts, calib_cache)
        print(f"Saved codeword counts → {calib_cache}")

    # Statistics
    n_active    = (counts > 0).sum().item()
    n_unused    = VOCAB_SIZE - n_active
    mask_tok_id = int(counts.argmin())

    print(
        f"\nCodeword statistics\n"
        f"  Vocabulary size : {VOCAB_SIZE}\n"
        f"  Active          : {n_active}  ({100*n_active/VOCAB_SIZE:.1f}%)\n"
        f"  Unused          : {n_unused}\n"
        f"  MASK_TOKEN id   : {mask_tok_id}  "
        f"(occurred {counts[mask_tok_id].item()} times)\n"
    )

    least = counts.topk(10, largest=False)
    print("  10 least-used codewords:")
    for rank, (idx, cnt) in enumerate(
            zip(least.indices.tolist(), least.values.tolist()), start=1):
        marker = "  ← MASK_TOKEN" if idx == mask_tok_id else ""
        print(f"    #{rank:2d}  id={idx:5d}  count={cnt}{marker}")

    if args.calibrate_only:
        return

    # ── build model ────────────────────────────────────────────────────────
    model = MaskedDALLEVAE(encoder, decoder, mask_token_id=mask_tok_id)

    # Restore weights if resuming
    if args.resume:
        print(f"\nLoading checkpoint {args.resume} …")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.mask_token_id = ckpt.get("mask_token_id", mask_tok_id)
        print(f"  Resumed from epoch {ckpt.get('epoch', '?')}")

    model.to(device)

    # ── evaluation-only branch ────────────────────────────────────────────
    if args.eval:
        print(f"\nRunning evaluation on validation set …")
        val_loader = get_loader("test", args.batch_size, args.num_workers)
        all_stats  = run_eval_suite(model, val_loader, device,
                                     mask_ratio=args.mask_ratio,
                                     max_batches=args.eval_batches)
        out = Path(args.output_dir) / "eval_stats.json"
        with open(out, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nEval stats saved → {out}")
        return

    # ── freeze layers ──────────────────────────────────────────────────────
    apply_freeze_strategy(model)
    trainable, total = _count_params(model)
    print(
        f"\nParameter budget\n"
        f"  Trainable : {trainable:>12,}  ({100*trainable/total:.2f}%)\n"
        f"  Frozen    : {total-trainable:>12,}\n"
        f"  Total     : {total:>12,}\n"
    )

    # ── fine-tune ──────────────────────────────────────────────────────────
    print(
        f"Fine-tuning for {args.epochs} epochs\n"
        f"  mask_ratio={args.mask_ratio}  batch={args.batch_size}  lr={args.lr}\n"
    )
    train(args, model, device)


if __name__ == "__main__":
    main()
