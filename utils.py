import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np
import pandas as pd

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess_image(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).cuda()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def annotation_change(annotation_path: str, output_path: str = None) -> pd.DataFrame:
    """
    기존 annotations.csv 파일에서 이전 10개 프레임의 left/right 속도를 컬럼으로 추가하여 반환합니다.
    필요한 경우, 수정된 데이터프레임을 새로운 CSV 파일로 저장할 수 있습니다.
    """
    # 1. CSV 읽기 (헤더가 없다는 가정하에)
    df = pd.read_csv(annotation_path, header=None, names=["frame_name", "current_left", "current_right"])

    # 2. 이전 10개의 left/right 시퀀스를 shift로 생성
    for i in range(1, 11):
        df[f"prev_left{i}"]  = df["current_left"].shift(i)
        df[f"prev_right{i}"] = df["current_right"].shift(i)

    # 3. NaN 값(처음 10개 행의 부족 데이터)을 0으로 채우기
    df.fillna(0, inplace=True)

    # 4. 컬럼 순서 재배치
    col_order = ["frame_name"]
    for i in range(10, 0, -1):
        col_order += [f"prev_left{i}", f"prev_right{i}"]
    col_order += ["current_left", "current_right"]

    df = df[col_order]

    # 5. 필요 시 CSV로 저장
    if output_path is not None:
        df.to_csv(output_path, index=False,header=False)

    return df

if __name__ == "__main__":
    annotation_change(
        "./datasets/train/annotations.csv",
        "./datasets/train/annotations_with_prev10.csv"
    )
    annotation_change(
        "./datasets/valid/annotations.csv",
        "./datasets/valid/annotations_with_prev10.csv"
    )