"""ONNX export + FP16/INT8 TRT engine build for the fine-tuned dVAE decoder.

Pipeline
========

    best_model.pt  (MaskedDALLEVAE checkpoint)
        -> dec.onnx           input  : (1, 8192, H', W') soft codewords
                              output : (1, 6, H, W)      logit-Laplace stats
        -> dec_fp16.trt       pure FP16
        -> dec_int8.trt       INT8 with FP16 fallback, calibrated with
                              realistic mixed inputs (one-hot for
                              "transmitted" positions, softmax(mask_token)
                              for "masked-out" positions — same distribution
                              the C++ subscriber feeds at inference time).

Also emits ``mask_token.bin`` — softmax(mask_token) as a flat [V] float32
binary. The C++ subscriber writes this distribution into every spatial
column of the decoder input that the publisher did NOT transmit, exactly
mirroring MaskedDALLEVAE.apply_mask during fine-tuning. This replaces the
old encoder-on-uniform-gray ``mask_codes.bin`` per-position one-hot.

Why FP16 fallback for INT8. The first 1x1 conv of the decoder projects the
8192-channel one-hot input to 128 channels — effectively a learnt codebook
lookup. INT8 quantisation on that layer noticeably degrades reconstructions;
FP16 fallback lets TRT keep precision-sensitive layers out of INT8.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tensorrt as trt

from dall_e import Encoder, load_model
from dataset import OpenImagesV7, CropConfig, download_openimages_v7

from finetune_mask_vae import MaskedDALLEVAE, load_finetuned_model

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

VOCAB_SIZE = 8192


def save_mask_token(model: MaskedDALLEVAE, out_path: Path) -> None:
    """Save softmax(mask_token) as a flat [V] float32 binary.

    Layout: V * float32, little-endian (host order on x86/aarch64).
    The C++ subscriber memcpys this into a [V] GPU tensor and broadcasts
    it across non-transmitted spatial columns of the decoder input.
    """
    dist = model.mask_token_distribution().detach().cpu().to(torch.float32)
    arr = dist.numpy()
    assert arr.shape == (VOCAB_SIZE,), arr.shape
    out_path.write_bytes(arr.tobytes())
    logging.info(
        f"wrote mask token: {out_path} ({arr.size} float32 values, "
        f"{out_path.stat().st_size} bytes, sum={arr.sum():.6f}, "
        f"max={arr.max():.6f} at id={int(arr.argmax())})"
    )


def export_decoder_onnx(model: MaskedDALLEVAE, onnx_path: Path,
                        h_prime: int, w_prime: int,
                        device: torch.device) -> None:
    """Export only the decoder submodule of the fine-tuned MaskedDALLEVAE."""
    dec = model.decoder
    dec.eval()

    # Export conv weights as plain FP32 initializers. DALL-E Conv2d defaults to
    # use_float16=True, so forward does self.w.half() — which becomes a
    # Cast(weight)->fp16 node in ONNX. TRT then routes the conv through its
    # dynamic-weight path and aborts parsing:
    #   UNSUPPORTED_NODE (convMultiInput): checkSpatialDims(...) assertion failed.
    # TRT picks per-layer precision from the FP16/INT8 builder flags, so the
    # graph itself must stay full-precision.
    for m in dec.modules():
        if hasattr(m, "use_float16"):
            m.use_float16 = False

    # A realistic dummy: one-hot at every spatial position. Matches
    # what the C++ subscriber feeds for "transmitted" positions and is
    # also valid (sums to 1 per column) for the masked-out variant.
    dummy = torch.zeros(1, VOCAB_SIZE, h_prime, w_prime, device=device)
    idx = torch.randint(0, VOCAB_SIZE, (h_prime * w_prime,), device=device)
    dummy.view(VOCAB_SIZE, -1).scatter_(0, idx.unsqueeze(0), 1.0)

    torch.onnx.export(
        dec, dummy, str(onnx_path),
        export_params=True,
        input_names=["codewords_one_hot"],
        output_names=["x_stats"],
        dynamic_axes=None,
        opset_version=17,
    )
    logging.info(f"wrote ONNX: {onnx_path} (input {h_prime}x{w_prime})")


class MixedInputCalibrator(trt.IInt8EntropyCalibrator2):
    """Streams (B, V, H', W') decoder-input tensors to TRT for INT8 calibration.

    Each batch reproduces what the decoder sees in production:

        * encoder argmax → one-hot at "transmitted" spatial positions
        * softmax(mask_token) at "masked-out" positions
        * the mask is sampled uniformly at random per-image with ratio
          ``mask_ratio`` (matches the fine-tuning distribution).

    The batch is staged through a single persistent CUDA buffer that the
    calibrator hands back to TRT by raw pointer.
    """

    def __init__(self, dataset, encoder: Encoder,
                 mask_token_dist: torch.Tensor,
                 cache_path: Path,
                 batch_size: int, num_batches: int,
                 h_prime: int, w_prime: int,
                 mask_ratio: float,
                 device: torch.device):
        super().__init__()
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.encoder = encoder
        self.h_prime = h_prime
        self.w_prime = w_prime
        self.mask_ratio = mask_ratio

        # mask_token_dist: [V] float32. We'll broadcast into masked columns.
        self.mask_token_dist = mask_token_dist.to(device=device,
                                                  dtype=torch.float32)

        self.buffer = torch.empty(
            (batch_size, VOCAB_SIZE, h_prime, w_prime),
            dtype=torch.float32, device=device,
        )

        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        self.iter = iter(self.loader)
        self.batches_used = 0

    def get_batch_size(self) -> int:
        return self.batch_size

    @torch.no_grad()
    def get_batch(self, names):
        if self.batches_used >= self.num_batches:
            return None
        try:
            x = next(self.iter)
        except StopIteration:
            return None
        x = x.to(self.device, non_blocking=True)
        b = x.shape[0]

        codes = torch.argmax(self.encoder(x), dim=1)                 # (B, H', W')
        one_hot = F.one_hot(codes, num_classes=VOCAB_SIZE)           # (B, H', W', V)
        one_hot = one_hot.permute(0, 3, 1, 2).to(torch.float32)      # (B, V, H', W')

        # Random per-position mask with fraction self.mask_ratio set True.
        # True == "masked out" → replace one-hot with mask_token_dist.
        rand = torch.rand(b, self.h_prime, self.w_prime, device=self.device)
        mask = (rand < self.mask_ratio).unsqueeze(1)                 # (B, 1, H', W')
        mt = self.mask_token_dist.view(1, VOCAB_SIZE, 1, 1)
        mixed = torch.where(mask, mt.expand_as(one_hot), one_hot)

        self.buffer[:b].copy_(mixed)
        if b < self.batch_size:
            self.buffer[b:].zero_()
        self.batches_used += 1
        if self.batches_used % 8 == 0:
            logging.info(
                f"calibration batch {self.batches_used}/{self.num_batches}"
            )
        return [int(self.buffer.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            logging.info(f"reusing cache: {self.cache_path}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)
        logging.info(
            f"wrote cache: {self.cache_path} ({len(cache)} bytes)"
        )


def _set_workspace(config, gib: int) -> None:
    try:
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, gib * (1 << 30)
        )
    except AttributeError:
        config.max_workspace_size = gib * (1 << 30)


def build_engine(onnx_path: Path, engine_path: Path, *,
                  fp16: bool, int8: bool, calibrator,
                  workspace_gib: int) -> None:
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logging.error(parser.get_error(i))
            raise RuntimeError(f"failed to parse {onnx_path}")

    config = builder.create_builder_config()
    _set_workspace(config, workspace_gib)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            raise RuntimeError("INT8 build requested but no calibrator provided")
        config.int8_calibrator = calibrator

    label = "INT8+FP16" if int8 else ("FP16" if fp16 else "FP32")
    logging.info(f"building {label} decoder engine…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed — check TRT logs above")
    engine_bytes = bytes(serialized)
    engine_path.write_bytes(engine_bytes)
    logging.info(
        f"wrote engine: {engine_path} "
        f"({len(engine_bytes) / 1e6:.1f} MB, {label})"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",
                   default="/home/leonard/arc_ws/src/agentic/checkpoints/best_model.pt",
                   help="Fine-tuned MaskedDALLEVAE checkpoint (best_model.pt)")
    p.add_argument("--encoder-pkl",
                   default="/home/leonard/arc_ws/src/agentic/checkpoints/encoder.pkl")
    p.add_argument("--decoder-pkl",
                   default="/home/leonard/arc_ws/src/agentic/checkpoints/decoder.pkl",
                   help="Pretrained decoder.pkl used only as a scaffold to "
                        "instantiate the Decoder module before loading the "
                        "fine-tuned state dict.")
    p.add_argument("--data-root", default="/mnt/hdd/dataset/OpenImagesV7",
                   help="OpenImagesV7 root. If empty, fiftyone downloads the "
                        "validation split here on first run.")
    p.add_argument("--calib-split", default="validation",
                   choices=["train", "validation", "test"])
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on images fetched by fiftyone (None = full split)")
    p.add_argument("--out-dir", default="/home/leonard/arc_ws/src")
    p.add_argument("--img-h", type=int, default=240,
                   help="Image height in pixels (must be multiple of 8). "
                        "Decoder input is (1, V, img-h/8, img-w/8).")
    p.add_argument("--img-w", type=int, default=320,
                   help="Image width in pixels (must be multiple of 8).")
    p.add_argument("--name-suffix", default="",
                   help="Suffix injected before file extensions: "
                        "dec{suffix}.onnx, dec_fp16{suffix}.trt, "
                        "dec_int8{suffix}.trt, dec{suffix}.cache. "
                        "Use e.g. '_63' for the 640x320 build. The "
                        "mask_token.bin is shape-independent (resolution-"
                        "agnostic) and is always written without a suffix.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=32)
    p.add_argument("--mask-ratio", type=float, default=0.5,
                   help="Fraction of spatial positions replaced with "
                        "softmax(mask_token) during INT8 calibration, "
                        "matching the fine-tuning distribution.")
    p.add_argument("--workspace-gib", type=int, default=4)
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument("--skip-int8", action="store_true")
    p.add_argument("--skip-onnx", action="store_true",
                   help="reuse an existing ONNX file in out-dir")
    p.add_argument("--skip-mask-token", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")

    img_h, img_w = args.img_h, args.img_w
    if img_h % 8 != 0 or img_w % 8 != 0:
        raise SystemExit(
            f"img-h ({img_h}) and img-w ({img_w}) must both be multiples of 8"
        )
    h_prime, w_prime = img_h // 8, img_w // 8

    sfx = args.name_suffix
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"dec{sfx}.onnx"
    fp16_path = out_dir / f"dec_fp16{sfx}.trt"
    int8_path = out_dir / f"dec_int8{sfx}.trt"
    cache_path = out_dir / f"dec{sfx}.cache"
    mask_path = out_dir / "mask_token.bin"

    logging.info(f"loading fine-tuned model from {args.checkpoint}")
    model = load_finetuned_model(
        args.checkpoint, args.encoder_pkl, args.decoder_pkl, device,
    )

    if not args.skip_mask_token:
        save_mask_token(model, mask_path)

    if not args.skip_onnx:
        export_decoder_onnx(model, onnx_path, h_prime, w_prime, device)
    elif not onnx_path.exists():
        raise FileNotFoundError(f"--skip-onnx set but {onnx_path} doesn't exist")

    if not args.skip_fp16:
        build_engine(onnx_path, fp16_path,
                     fp16=True, int8=False, calibrator=None,
                     workspace_gib=args.workspace_gib)

    if not args.skip_int8:
        data_root = Path(args.data_root).expanduser().resolve()
        data_root.mkdir(parents=True, exist_ok=True)
        val_root = download_openimages_v7(
            str(data_root), args.calib_split, max_samples=args.max_samples,
        )

        dataset = OpenImagesV7(
            val_root,
            CropConfig(size=(img_h, img_w),
                       random_crop=False, horizontal_flip=False),
            training=False,
        )
        total = min(args.num_batches * args.batch_size, len(dataset))
        logging.info(
            f"calibration: {total} images from {val_root} at {img_h}x{img_w} "
            f"(mask_ratio={args.mask_ratio})"
        )
        calibrator = MixedInputCalibrator(
            dataset, model.encoder,
            mask_token_dist=model.mask_token_distribution().detach(),
            cache_path=cache_path,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            h_prime=h_prime, w_prime=w_prime,
            mask_ratio=args.mask_ratio,
            device=device,
        )
        build_engine(onnx_path, int8_path,
                     fp16=True, int8=True, calibrator=calibrator,
                     workspace_gib=args.workspace_gib)

    print()
    print("=" * 64)
    print(f" mask token: {mask_path}")
    print(f" ONNX:       {onnx_path}")
    if not args.skip_fp16:
        print(f" FP16:       {fp16_path}")
    if not args.skip_int8:
        print(f" INT8:       {int8_path}")
        print(f" cache:      {cache_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
