import csv
import torch
from sklearn.decomposition import PCA
from torchvision import transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from autopilot_model import E2E_CNN, E2E_CRNN

def load_models(device: str):
    cnn = E2E_CNN(pretrained=False).to(device)
    crnn = E2E_CRNN(pretrained=False).to(device)
    cnn.load_state_dict(torch.load("./checkpoint/last_model_cnn.pt",  weights_only=True))
    crnn.load_state_dict(torch.load("./checkpoint/last_model_rnn.pt", weights_only=True))
    cnn.eval();  crnn.eval()
    return cnn, crnn

def load_speed_sequences(csv_path: Path) -> dict:
    seq_map = {}
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        for parts in reader:
            name = parts[0].strip()
            prev_seq = [[float(parts[i]), float(parts[i+1])] for i in range(1,21,2)]
            gt_left, gt_right = float(parts[21]), float(parts[22])
            seq_map[name] = {"prev_seq": prev_seq, "gt": (gt_left, gt_right)}
    return seq_map

def preprocess(image_bgr: np.ndarray):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224,224)).astype(np.float32)/255.0
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return tfm(image_resized).unsqueeze(0), image_resized

def visualize_pca(feat: torch.Tensor, n_components: int = 1):
    C,H,W = feat.shape
    X = feat.reshape(C, -1).permute(1,0).cpu().numpy()  # (H*W, C)
    pca = PCA(n_components=n_components)
    X_p = pca.fit_transform(X)
    X_p -= X_p.min(axis=0);  X_p /= (X_p.max(axis=0)+1e-6)
    out = X_p.reshape(H, W, n_components)
    return out[:,:,0] if n_components==1 else out

def main():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    valid_dir = Path("datasets/valid")
    save_dir  = Path("valid_pca")
    save_dir.mkdir(parents=True, exist_ok=True)

    seq_map = load_speed_sequences(valid_dir/"annotations_with_prev10.csv")
    cnn, crnn = load_models(device)

    for name, data in seq_map.items():
        img_path = valid_dir/f"{name}.jpg"
        if not img_path.is_file(): continue

        # 전처리
        bgr = cv2.imread(str(img_path))
        inp_t, img_rgb = preprocess(bgr)
        inp_t = inp_t.to(device)
        speed_seq = torch.tensor([data["prev_seq"]], dtype=torch.float32, device=device)

        # CNN feature
        with torch.no_grad():
            _ = cnn(inp_t)
            feat_cnn = cnn.feature_map.squeeze(0)  # (C,H,W)
            print(feat_cnn.shape)
        pca_cnn = visualize_pca(feat_cnn, n_components=1)

        # CRNN feature
        with torch.no_grad():
            _ = crnn(inp_t, speed_seq)
            feat_crnn = crnn.feature_map.squeeze(0)
        pca_crnn = visualize_pca(feat_crnn, n_components=1)

        # 리사이즈
        rgb_resized = cv2.resize(img_rgb, (pca_cnn.shape[1], pca_cnn.shape[0]))

        # 시각화: 1×3
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        # 1) Input Image
        ax[0].imshow(rgb_resized)
        ax[0].set_title("Input Image")
        ax[0].axis("off")
        # 2) CNN PCA
        im1 = ax[1].imshow(pca_cnn, cmap="viridis")
        ax[1].set_title("CNN Feature PCA")
        ax[1].axis("off")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        # 3) CRNN PCA
        im2 = ax[2].imshow(pca_crnn, cmap="viridis")
        ax[2].set_title("CRNN Feature PCA")
        ax[2].axis("off")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        # 전체 suptitle에 GT 표시
        gt_s, gt_t = data["gt"]
        fig.suptitle(f"{name}  GT Steering={gt_s:.3f}, Throttle={gt_t:.3f}", fontsize=14, y=1.02)
        plt.tight_layout()
        out_path = save_dir/f"{name}_pca_comparison.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved → {out_path}")

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    main()
