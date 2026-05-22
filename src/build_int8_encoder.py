"""INT8 calibration + engine build for the DALL-E encoder.

Target hardware
===============
Jetson Orin Nano (sm_87). Run this script on the Jetson itself: TensorRT
serialises engines for the GPU it was built on, so an engine built on a
desktop A5000 will not load on Orin. The calibration cache (`enc.cache`) is
hardware-portable, however, so you can produce it on any CUDA box with the
same TRT major.minor and copy it over to skip recalibration.

Pipeline
========

    enc.onnx  (1, 3, 240, 320)                  exported by exporting.ipynb;
                                                output: logits (1, 8192, 30, 40)
        -> TRT IInt8EntropyCalibrator2          (real images, map_pixels)
        -> portable INT8 calibration cache      (enc.cache)
        -> INT8 TRT engine                      (enc_int8.trt)

Why FP16 fallback. The final 1x1 conv projects to V=8192 channels. Whatever
quantisation noise that layer adds is multiplied by the softmax-style spread
across vocab logits, which the host-side argmax picks the top of. INT8 on
the vocab projection visibly perturbs token agreement vs the PyTorch
reference encoder; FP16 fallback lets TRT keep precision-sensitive layers
out of INT8.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Desktop dev box: point at the local TRT install. On Jetson the system TRT
# is already on the loader path, so this env-default is harmless.
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    "/home/leonard/TensorRT-8.6.1.6/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
)

import torch
from torch.utils.data import DataLoader

import tensorrt as trt

from dataset import OpenImagesV7, CropConfig

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Streams (B, 3, H, W) fp32 images already in `map_pixels` space.

    `OpenImagesV7` returns map_pixels'd tensors on CPU; we stage them through
    a single persistent fp32 CUDA buffer that the calibrator hands back to
    TRT by raw pointer."""

    def __init__(self, dataset, cache_path: Path, batch_size: int,
                 image_hw: tuple[int, int], num_batches: int,
                 device: torch.device):
        super().__init__()
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        H, W = image_hw
        self.num_batches = num_batches
        self.device = device

        self.buffer = torch.empty(
            (batch_size, 3, H, W), dtype=torch.float32, device=device,
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
        self.buffer[:b].copy_(x)
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


def build_int8_engine(onnx_path: Path, calibrator, engine_path: Path,
                       workspace_gib: int, fp16_fallback: bool) -> None:
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
    config.set_flag(trt.BuilderFlag.INT8)
    if fp16_fallback:
        config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    logging.info("building INT8 encoder engine (drives the calibrator)…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed — check TRT logs above")
    engine_bytes = bytes(serialized)
    engine_path.write_bytes(engine_bytes)
    logging.info(
        f"wrote engine: {engine_path} "
        f"({len(engine_bytes) / 1e6:.1f} MB)"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--onnx",   default="/home/leonard/arc_ws/src/enc.onnx")
    p.add_argument("--data-root", default="/home/leonard/dvae/data")
    p.add_argument("--cache",  default="/home/leonard/arc_ws/src/enc.cache")
    p.add_argument("--engine", default="/home/leonard/arc_ws/src/enc_int8.trt")
    p.add_argument("--image-size", default="240x320",
                    help="HxW image size — must match ONNX's input dims")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=64)
    p.add_argument("--workspace-gib", type=int, default=2,
                    help="Default 2 GiB suits Orin Nano's 8 GB unified memory; "
                         "bump on a desktop box if the builder asks for more.")
    p.add_argument("--no-fp16-fallback", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    H, W = (int(s) for s in args.image_size.split("x"))
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    val_root = Path(args.data_root) / "validation"
    if not val_root.exists():
        val_root = Path(args.data_root)
    if not val_root.exists():
        raise FileNotFoundError(f"validation images not found at {val_root}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")

    dataset = OpenImagesV7(
        val_root,
        CropConfig(size=(H, W), random_crop=False, horizontal_flip=False),
        training=False,
    )
    total = min(args.num_batches * args.batch_size, len(dataset))
    logging.info(
        f"calibration: {total} images from {val_root} at {H}x{W}"
    )

    calibrator = ImageCalibrator(
        dataset,
        cache_path=Path(args.cache),
        batch_size=args.batch_size,
        image_hw=(H, W),
        num_batches=args.num_batches,
        device=device,
    )

    build_int8_engine(
        onnx_path, calibrator, Path(args.engine),
        workspace_gib=args.workspace_gib,
        fp16_fallback=not args.no_fp16_fallback,
    )

    print()
    print("=" * 64)
    print(f" cache:  {args.cache}")
    print(f" engine: {args.engine}")
    print("=" * 64)


if __name__ == "__main__":
    main()
