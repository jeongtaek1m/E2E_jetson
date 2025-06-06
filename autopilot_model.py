import torch
import torchvision
import torch.nn as nn

OUTPUT_SIZE = 2
DROPOUT_PROB = 0.4
RNN_HIDDEN_SIZE = 64  # RNN의 은닉 상태 차원 (하이퍼파라미터)
SEQ_LEN = 10          # 과거 속도 시퀀스 길이


class E2E_CNN(nn.Module):
    """ResNet‑18 backbone with a small MLP head.

    Parameters
    ----------
    output_size : int
        Dimension of the final regression output (e.g. 2 for *steer*, *throttle*).
    dropout_prob : float, optional
        Dropout probability applied between the fully‑connected layers.
    pretrained : bool, optional
        If *True*, load ImageNet‑1K weights for the backbone.
    """

    def __init__(
        self,
        output_size: int = 2,
        dropout_prob: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = torchvision.models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_size),
        )

        # Feature‑extractor is **created lazily** the first time it is needed.
        # self._feature_extractor: nn.Module | None = None

    # --------------------------------------------------------------------- #
    #  Forward                                                                #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Inference – maps an RGB image tensor to actuator commands."""
        return self.backbone(x)


class E2E_CRNN(nn.Module):
    """
    - x: (B, 3, H, W) 형태의 RGB 이미지 배치
    - speed_seq: (B, seq_len=10, 2) 형태의 이전 left/right 속도 시퀀스 배치
    """
    def __init__(
        self,
        output_size: int = OUTPUT_SIZE,
        dropout_prob: float = DROPOUT_PROB,
        pretrained: bool = True,
        rnn_hidden_size: int = RNN_HIDDEN_SIZE,
        seq_len: int = SEQ_LEN,
    ) -> None:
        super().__init__()

        # 1) CNN(ResNet-18) 백본: fc를 Identity로 바꿔서 avgpool 후 512차원 피처만 추출
        self.cnn = torchvision.models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.cnn.fc = nn.Identity()  # fc층 제거

        # 2) RNN 부분: nn.RNN을 사용 (batch_first=True)
        #    입력 차원(input_size)=2(left, right), hidden_size=RNN_HIDDEN_SIZE, num_layers=1
        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.rnn_hidden_size = rnn_hidden_size

        # 3) CNN(512) + RNN(hidden_size) → 합친 피처를 FC 레이어로 예측
        combined_dim = 512 + rnn_hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, output_size),
        )

    def forward(self, x: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 형태의 현재 프레임
            speed_seq: (B, seq_len=10, 2) 형태의 과거 속도 시퀀스 (left, right)
        Returns:
            (B, 2) 형태의 예측된 (left, right) 속도
        """
        # 1) CNN으로부터 512차원 피처 추출
        cnn_feat = self.cnn(x)  # shape = (B, 512)

        # 2) RNN에 과거 속도 시퀀스 입력 → h_n에서 마지막 레이어 은닉 상태 추출
        # speed_seq shape = (B, 10, 2)
        rnn_out, h_n = self.rnn(speed_seq)
        # h_n shape = (num_layers=1, B, rnn_hidden_size)
        last_h = h_n[-1]  # shape = (B, rnn_hidden_size)

        # 3) CNN 피처와 RNN 마지막 은닉 상태를 이어 붙임
        combined = torch.cat([cnn_feat, last_h], dim=1)  # shape = (B, 512 + rnn_hidden_size)

        # 4) FC 레이어로 통과시켜 최종 (left, right) 예측
        output = self.fc(combined)  # shape = (B, 2)
        return output