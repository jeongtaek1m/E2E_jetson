import torch
import torchvision
import torch.nn as nn

OUTPUT_SIZE = 2
DROPOUT_PROB = 0.4
RNN_HIDDEN_SIZE = 64  
SEQ_LEN = 10       

class E2E_CNN(nn.Module):
    def __init__(self, output_size=2, dropout_prob=0.3, pretrained=False, freeze_front=False):
        super().__init__()
        self.backbone = torchvision.models.resnet18(
            weights=None if not pretrained else "IMAGENET1K_V1"
        )
        if freeze_front:
            for m in [self.backbone.conv1, self.backbone.bn1,
                      self.backbone.layer1, self.backbone.layer2]:
                for p in m.parameters():
                    p.requires_grad = False
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_f, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_size),
        )
    def forward(self, x):
        return self.backbone(x)

class E2E_CRNN(nn.Module):
    def __init__(self, output_size=2, dropout_prob=0.4,
                 pretrained=False, rnn_hidden_size=64, seq_len=10, freeze_front=False):
        super().__init__()
        self.cnn = torchvision.models.resnet18(
            weights=None if not pretrained else "IMAGENET1K_V1"
        )
        if freeze_front:
            for m in [self.cnn.conv1, self.cnn.bn1,
                      self.cnn.layer1, self.cnn.layer2]:
                for p in m.parameters():
                    p.requires_grad = False
        self.cnn.fc = nn.Identity()
        self.rnn = nn.RNN(input_size=2, hidden_size=rnn_hidden_size,
                          num_layers=1, batch_first=True)
        combined = 512 + rnn_hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, output_size),
        )
    def forward(self, x, seq):
        f = self.cnn(x)
        _, h = self.rnn(seq)
        h = h[-1]
        return self.fc(torch.cat([f, h], dim=1))

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    cnn   = E2E_CNN(pretrained=False, freeze_front=False)
    crnn  = E2E_CRNN(pretrained=False, freeze_front=False)

    for m in [cnn, crnn]:
        tot, tr = count_params(m)
        print(f"{m.__class__.__name__:10s} ▶ total: {tot:,}개, trainable: {tr:,}개")
