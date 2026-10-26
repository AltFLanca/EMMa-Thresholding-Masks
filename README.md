# EMMa-Thresholding-Masks

Generate foreground masks for EMMa product images using grayscale inverse thresholding followed by morphological opening to remove noise. Currently, only the first product image associated with each ASIN is processed. Dataset access and augmentation storage are handled through [EMMa-Wrapper](https://github.com/AltFLanca/EMMa-Wrapper).

## Setup

Install the project dependencies using [uv](https://docs.astral.sh/uv/). From the repository root, run:

```bash
uv sync --locked
```

A prepared EMMa dataset is required. The dataset is not included in this repository.

## Repository Structure

- `src/test_single.ipynb` - Notebook for testing individual samples.
- `src/test_inference.ipynb` - Notebook for testing batched inference.
- `src/inference.py` - Generates foreground masks using thresholding and morphological operations.
- `src/valid_output_filtering.py` - Produces a list of valid ASINs from the generated augmentations.
- `src/visualize_augmentation_output.ipynb` - Visualizes generated masks for inspection.

## Generate Augmentations

Mask generation requires an `.npy` file containing the ASINs to process, provided through the `--asins` argument.

The thresholding and morphological cleanup operations can be configured using the following arguments:

- `--threshold` - Grayscale threshold used for inverse binary thresholding.
- `--kernel-size` - Kernel size used for morphological opening.
- `--open-iterations` - Number of morphological opening iterations.

Run inference with:

```bash
uv run src/inference.py \
  --dataset-root /path/to/EMMa \
  --asins /path/to/augmented_asins.npy \
  --output /path/to/EMMa-Thresholding-Masks/masks.h5 \
  --threshold 253 \
  --kernel-size 3 \
  --open-iterations 2
```

Generated binary masks are stored in an HDF5 file at the specified output path, one per ASIN. To inspect the generated masks, use [visualize_augmentation_output.ipynb](src/visualize_augmentation_output.ipynb).

**Note:** Inference automatically skips ASINs already present in the output file. When changing thresholding parameters, use a new output path to avoid mixing results generated with different configurations.

## Validate Augmentations

The [valid_output_filtering.py](src/valid_output_filtering.py) script processes the generated augmentations and produces a `valid_asins.npy` file in the same directory as the input HDF5 file.

Run validation with:

```bash
uv run src/valid_output_filtering.py \
  --dataset-root /path/to/EMMa \
  --augmentation-file /path/to/EMMa-Thresholding-Masks/masks.h5
```

Because the output is always named `valid_asins.npy`, validating multiple HDF5 files in the same directory overwrites the previous list.

**Note:** The current implementation does not apply any filtering criteria. All processed ASINs are considered valid.