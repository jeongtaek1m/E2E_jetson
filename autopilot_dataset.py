from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from typing import List, Tuple
from typing import Union
class AutopilotDataset(Dataset):
    """Simple (image, action) dataset for end‑to‑end driving.

    Args:
        directory: Path to the dataset root containing ``annotations.csv`` and images.
        frame_size: Final (width == height) resolution fed to the network.
        transform: Extra deterministic transforms to run after *all* random ones.
        random_noise: Inject salt‑and‑pepper noise with 50 % probability.
        random_blur: Apply Gaussian blur with 50 % probability.
        random_horizontal_flip: Horizontally flip image **and** swap left/right labels with 50 % probability.
        random_color_jitter: Color jitter (0.3 probability) — brightness/contrast/hue/sat ±0.25.
        keep_images_in_ram: If ``True`` preload and keep decoded ``PIL.Image`` objects in memory.
    """

    def __init__(
        self,
        directory: str | Path,
        frame_size: int,
        *,
        transform: T.Compose | None = None,
        random_noise: bool = False,
        random_blur: bool = False,
        random_horizontal_flip: bool = False,
        random_color_jitter: bool = False,
        keep_images_in_ram: bool = False,
        use_rnn: bool = False,
    ) -> None:
        super().__init__()

        self.root = Path(directory)
        self.frame_size = frame_size
        self.extra_transform = transform
        self.random_noise = random_noise
        self.random_blur = random_blur
        self.random_horizontal_flip = random_horizontal_flip
        self.keep_images_in_ram = keep_images_in_ram
        self.use_rnn = use_rnn

        # ------------------------------------------------------------------
        #  Random transforms configured once here, re‑used in __getitem__.
        # ------------------------------------------------------------------
        self.color_jitter: T.RandomApply | None = None
        if random_color_jitter:
            self.color_jitter = T.RandomApply(
                [T.ColorJitter(brightness=0.25, contrast=0.25,
                                saturation=0.25, hue=0.25)],
                p=0.3,
            )

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
        ])

        # ------------------------------------------------------------------
        #  Build in‑memory index  ▸  self.samples = List[(name, img_or_path, l, r)]
        # ------------------------------------------------------------------
        self.samples: List[Tuple[str, str | Image.Image, float, float]] = []
        if use_rnn:
            ann_path = self.root / "annotations_with_prev10.csv"
            if not ann_path.is_file():
                raise FileNotFoundError(f"RNN annotations file not found: {ann_path}")

            with ann_path.open() as fh:
                for line in fh:
                    parts = [f.strip() for f in line.split(",")]
                    # parts: [frame_name, prev_left10, prev_right10, ..., prev_left1, prev_right1, current_left, current_right]
                    name = parts[0]
                    img_path = self.root / f"{name}.jpg"
                    if not img_path.is_file() or img_path.stat().st_size == 0:
                        continue

                    # 20개 이전 속도 값 파싱
                    prev_seq: List[List[float]] = []
                    for idx in range(1, 21, 2):
                        left_val = float(parts[idx])
                        right_val = float(parts[idx + 1])
                        prev_seq.append([left_val, right_val])
                    # 이제 prev_seq = [[prev_left10, prev_right10], ..., [prev_left1, prev_right1]]

                    current_left = float(parts[21])
                    current_right = float(parts[22])

                    record_img: str | Image.Image
                    if keep_images_in_ram:
                        record_img = Image.open(img_path).convert("RGB")
                    else:
                        record_img = str(img_path)

                    self.samples.append((name, record_img, prev_seq, current_left, current_right))


        else:
            ann_path = self.root / "annotations.csv"
            with ann_path.open() as fh:
                for line in fh:
                    name, left, right = (f.strip() for f in line.split(","))
                    img_path = self.root / f"{name}.jpg"
                    if not img_path.is_file() or img_path.stat().st_size == 0:
                        continue  # skip missing / empty files

                    record_img: str | Image.Image
                    if keep_images_in_ram:
                        record_img = self._load_pil(img_path)
                    else:
                        record_img = str(img_path)

                    self.samples.append((name, record_img, float(left), float(right)))

        print(f"Loaded {len(self.samples):,} samples from {self.root}")

    # ---------------------------------------   ---------------------------
    #  Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.samples)

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        item = self.samples[index]
        name = item[0]
        img_or_path = item[1]

        # Lazily load if not pre‑cached
        img: Image.Image = img_or_path if isinstance(img_or_path, Image.Image) else self._load_pil(img_or_path)

        # --------------------------------------------------------------
        #  Random augmentations (50 % unless stated otherwise)
        # --------------------------------------------------------------
        if self.random_blur and np.random.rand() > 0.5:
            img = img.filter(ImageFilter.BLUR)

        if self.random_noise and np.random.rand() > 0.5:
            img = Image.fromarray(self._salt_and_pepper(np.array(img)))

        if self.random_horizontal_flip and np.random.rand() > 0.5:
            img = T.functional.hflip(img)
            # swap motor commands (left ↔ right) when flipped
            left, right = right, left

        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # Extra deterministic transforms (e.g., Cutout)
        if self.extra_transform is not None:
            img = self.extra_transform(img)

        # Final tensor conversion
        img_tensor = self.to_tensor(img)
        if not self.use_rnn:
            # item = (name, img_or_path, left, right)
            left, right = item[2], item[3]
            target = torch.tensor([left, right], dtype=torch.float32)
            return img_tensor, target
        else:
            # item = (name, img_or_path, prev_seq, current_left, current_right)
            prev_seq, current_left, current_right = item[2], item[3], item[4]
            speed_seq = torch.tensor(prev_seq, dtype=torch.float32)  # shape = (10, 2)
            target = torch.tensor([current_left, current_right], dtype=torch.float32)
            return img_tensor, speed_seq, target


    # ------------------------------------------------------------------
    #  Helper utilities
    # ------------------------------------------------------------------
    def _load_pil(self, path: str | Path) -> Image.Image:
        """Open → BGR→RGB → (optional) center‑crop → resize → PIL.Image."""
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(path)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        # arr = center_crop_square(arr)  # Uncomment if aspect‑ratio‑preserving crop desired
        arr = cv2.resize(arr, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
        return Image.fromarray(arr)

    @staticmethod
    def _salt_and_pepper(img: np.ndarray, amount: float = 0.1) -> np.ndarray:
        """Impose salt‑and‑pepper noise in‑place and return noisy image."""
        noisy = img.copy()
        n_total = img.size
        n_salt = int(np.ceil(amount * n_total * 0.5))
        n_pepper = n_salt

        # Salt  ≈ white pixels
        coords = [np.random.randint(0, i, n_salt) for i in img.shape]
        noisy[tuple(coords)] = 255

        # Pepper ≈ black pixels
        coords = [np.random.randint(0, i, n_pepper) for i in img.shape]
        noisy[tuple(coords)] = 0
        return noisy
