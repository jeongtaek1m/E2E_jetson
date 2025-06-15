import csv
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from autopilot_model import E2E_CNN, E2E_CRNN

# ────────────────────────────────────────────────────────────────────────────────
class _MultiInput:
    def __init__(self, *tensors):
        self.tensors = tensors
    def to(self, device):
        return _MultiInput(*[t.to(device) for t in self.tensors])
    def __iter__(self):
        return iter(self.tensors)
    @property
    def shape(self):
        return self.tensors[0].shape
    def __getattr__(self, name):
        return getattr(self.tensors[0], name)

class CRNNWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, inp):
        img, seq = tuple(inp)
        return self.base(img, seq)

class RegressionTarget:
    def __init__(self, idx):
        self.idx = idx
    def __call__(self, out):
        return out[:, self.idx] if out.dim() == 2 else out[self.idx]

def load_models(device: str):
    cnn = E2E_CNN(pretrained=False).to(device)
    crnn = E2E_CRNN(pretrained=False).to(device)
    cnn.load_state_dict(torch.load("./checkpoint/last_model_cnn.pt",  weights_only=True))
    crnn.load_state_dict(torch.load("./checkpoint/last_model_rnn.pt", weights_only=True))
    cnn.eval()
    crnn.eval()
    return cnn, crnn

def preprocess(image_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return tfm(image_rgb).unsqueeze(0), image_rgb  # (1,3,224,224), HWC numpy

def load_speed_sequences(csv_path: Path) -> dict:
    """
    CSV 포맷:
    frame_name,
    prev_left10, prev_right10, ..., prev_left1, prev_right1,
    current_left, current_right
    """
    seq_map = {}
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        for parts in reader:
            name = parts[0].strip()
            # 이전 10프레임 속도 시퀀스
            prev_seq = [[float(parts[i]), float(parts[i+1])] for i in range(1, 21, 2)]
            # GT 스티어/스로틀
            gt_left  = float(parts[21])
            gt_right = float(parts[22])
            seq_map[name] = {
                "prev_seq": prev_seq,
                "gt": (gt_left, gt_right)
            }
    return seq_map

def main():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    valid_dir = Path("datasets/valid")
    save_dir  = Path("valid_prediction")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 어노테이션 한 번 읽기
    ann_csv = valid_dir / "annotations_with_prev10.csv"
    seq_map = load_speed_sequences(ann_csv)

    # 2) 모델 & Grad-CAM 준비
    cnn, crnn = load_models(device)
    layer_cnn  = cnn.backbone.layer4[-1]
    layer_crnn = crnn.cnn.layer4[-1]
    cam_cnn    = GradCAM(model=cnn,                target_layers=[layer_cnn])
    cam_crnn   = GradCAM(model=CRNNWrapper(crnn),  target_layers=[layer_crnn])
    tgt_steer  = [RegressionTarget(0)]
    tgt_throt  = [RegressionTarget(1)]

    # 3) 각 프레임별로 처리
    for name, data in seq_map.items():
        prev_seq        = data["prev_seq"]
        gt_left, gt_right = data["gt"]

        img_path = valid_dir / f"{name}.jpg"
        if not img_path.is_file():
            continue

        # 실제 시퀀스를 tensor로
        speed_seq = torch.tensor([prev_seq], dtype=torch.float32, device=device)  # (1,10,2)

        # 이미지 로드 & 전처리
        bgr      = cv2.imread(str(img_path))
        inp_t, img_rgb = preprocess(bgr)
        inp_t    = inp_t.to(device)

        # 예측
        with torch.no_grad():
            pred_cnn  = cnn(inp_t)[0].cpu().tolist()
            pred_crnn = crnn(inp_t, speed_seq)[0].cpu().tolist()

        # Grad-CAM 맵 생성
        g_cs = cam_cnn (input_tensor=inp_t, targets=tgt_steer)[0]
        g_ct = cam_cnn (input_tensor=inp_t, targets=tgt_throt)[0]
        multi = _MultiInput(inp_t, speed_seq)
        g_rs = cam_crnn(input_tensor=multi,  targets=tgt_steer)[0]
        g_rt = cam_crnn(input_tensor=multi,  targets=tgt_throt)[0]

        # 히트맵 입히기
        h_cs = show_cam_on_image(img_rgb, g_cs, use_rgb=True)
        h_ct = show_cam_on_image(img_rgb, g_ct, use_rgb=True)
        h_rs = show_cam_on_image(img_rgb, g_rs, use_rgb=True)
        h_rt = show_cam_on_image(img_rgb, g_rt, use_rgb=True)

        # 2×2 그리드에 출력
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        for a, img, title in zip(
            ax.flatten(),
            [h_cs, h_ct, h_rs, h_rt],
            ["CNN • left", "CNN • right", "CRNN • left", "CRNN • right"]
        ):
            a.imshow(img)
            a.set_title(title)
            a.axis("off")

        # GT와 예측값을 suptitle에
        fig.suptitle(
            f"GT    (s={gt_left:.3f}, t={gt_right:.3f})\n"
            f"CNN   (s={pred_cnn[0]:.3f}, t={pred_cnn[1]:.3f}) │ "
            f"CRNN  (s={pred_crnn[0]:.3f}, t={pred_crnn[1]:.3f})",
            fontsize=16, y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 저장
        out_name = save_dir / f"{name}_gradcam.png"
        fig.savefig(out_name, dpi=300)
        plt.close(fig)
        print(f"saved → {out_name}")

if __name__ == "__main__":
    # deterministic하게 실행되도록 CuDNN 비활성화
    torch.backends.cudnn.enabled = False
    main()
