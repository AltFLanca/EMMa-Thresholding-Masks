from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Any

import cv2
import hdf5plugin
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from emma import EmmaDataset
from emma.augment import HDF5Source
from emma.dataloader import EmmaMapDataset, emma_collate

import utils.logging_ as logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the thresholding foreground segmentation over the EMMa dataset."
            "Stores one mask per object in an HDF5 augmentation file."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing the EMMa dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 file. Defaults to <dataset_root>/<augmentation-name>.h5.",
    )
    parser.add_argument(
        "--augmentation-name",
        default="thresholding_masks",
        help="Registered augmentation name.",
    )
    parser.add_argument(
        "--asins",
        type=Path,
        required=True,
        help="Path to a .npy file containing the ASINs to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor used only when num_workers > 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N pending ASINs.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=253,
        help="Grayscale threshold used before THRESH_BINARY_INV.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Square morphology kernel size.",
    )
    parser.add_argument(
        "--open-iterations",
        type=int,
        default=2,
        help="Number of morphology open iterations.",
    )
    return parser.parse_args()


def load_asins(path: Path) -> list[str]:
    asins = np.load(path, allow_pickle=False)
    return [str(asin) for asin in asins.reshape(-1)]


def make_transform(ds: EmmaDataset):
    def transform(asin: str, rec: dict) -> dict:
        image_dict = ds.get_images(rec)
        first_img_key, first_img_bytes = next(iter(image_dict.items()))
        first_img = Image.open(io.BytesIO(first_img_bytes)).convert("RGB")

        return {
            "image": np.array(first_img),
            "asin": asin,
            "image_key": first_img_key,
        }

    return transform


def thresholding_segmentation(
    image_rgb: np.ndarray,
    threshold: int,
    kernel_size: int,
    open_iterations: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    bin_img = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_OPEN,
        kernel,
        iterations=open_iterations,
    )
    return (bin_img > 127).astype(np.uint8, copy=False)


def make_writer(
    name: str,
    path: Path,
) -> HDF5Source:
    compression = hdf5plugin.Blosc(
        cname="lz4hc",
        clevel=5,
        shuffle=hdf5plugin.Blosc.SHUFFLE,
    )
    writer_kwargs: dict[str, Any] = {
        "mode": "a",
        "compression": compression,
        "shuffle": False,
        "fletcher32": False,
    }
    return HDF5Source(name, path, **writer_kwargs)


def missing_asins(writer: HDF5Source, asins: list[str]) -> list[str]:
    existing = set(writer.keys())
    return [asin for asin in asins if asin not in existing]


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    output_path = args.output or (dataset_root / f"{args.augmentation_name}.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving augmentation to {output_path}")

    ds = EmmaDataset(dataset_root)
    asins = load_asins(args.asins)
    print(f"Candidate ASINs: {len(asins)}")

    writer = make_writer(
        args.augmentation_name,
        output_path,
    )

    pending_asins = missing_asins(writer, asins)
    if args.limit is not None:
        pending_asins = pending_asins[: args.limit]
    print(f"Pending ASINs: {len(pending_asins)}")
    if not pending_asins:
        writer.close()
        print("Nothing to do.")
        return

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": emma_collate,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(
        EmmaMapDataset(ds, pending_asins, transform=make_transform(ds)),
        **loader_kwargs,
    )

    start = time.time()
    for batch_idx, batch in enumerate(loader):
        asin_batch = list(batch["asin"])
        image_batch = batch["image"]

        for asin, image in zip(asin_batch, image_batch, strict=True):
            mask = thresholding_segmentation(
                np.asarray(image),
                threshold=args.threshold,
                kernel_size=args.kernel_size,
                open_iterations=args.open_iterations,
            )
            writer.save(asin, mask)

        logging.progress_bar(batch_idx, len(loader), 1, 1)

    writer.close()
    print(f"Elapsed: {time.time() - start:.2f}s")
    print("Done")


if __name__ == "__main__":
    main()
