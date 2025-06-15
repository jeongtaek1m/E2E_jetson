import pandas as pd

# 파일 경로
path = "/home/jeong/gitclone/E2E_jetson/datasets/valid/annotations.csv"

# CSV 읽기 (헤더 없음)
df = pd.read_csv(path, header=None)

# 3번째 열(index 2)에 1.03 곱하기
df[2] = df[2] * 1.03

# 원본 파일에 덮어쓰기
df.to_csv("annotation.csv", header=False, index=False)







