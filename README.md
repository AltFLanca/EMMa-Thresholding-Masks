Inference run with command:

```
uv run src/inference.py --dataset-root /home/user/datasets/EMMa --asins /home/user/datasets/EMMa-ASINs/augmented_asins.npy --output /home/user/datasets/EMMa-Thresholding-Masks/masks.h5 --threshold 253 --kernel-size 3 --open-iterations 2
```

Then:

```
uv run src/valid_output_filtering.py --dataset-root /home/user/datasets/EMMa --augmentation-file /home/user/datasets/EMMa-Thresholding-Masks/masks.h5
```