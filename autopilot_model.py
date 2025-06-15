import torch
import torchvision
import torch.nn as nn
import wandb

OUTPUT_SIZE = 2
DROPOUT_PROB = 0.4
RNN_HIDDEN_SIZE = 64  
SEQ_LEN = 10       


class E2E_CNN(nn.Module):
    def __init__(
        self,
        output_size: int = 2,
        dropout_prob: float = 0.3,
        pretrained: bool = True,
        freeze_front: bool = False,
    ) -> None:
        
        super().__init__()

        self.backbone = torchvision.models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None
        )

        if freeze_front:
            # conv1, bn1, layer1, layer2 얼리기
            for module in [self.backbone.conv1, self.backbone.bn1,
                           self.backbone.layer1, self.backbone.layer2]:
                for p in module.parameters():
                    p.requires_grad = False


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.backbone(x)
        


class E2E_CRNN(nn.Module):

    def __init__(
        self,
        output_size: int = OUTPUT_SIZE,
        dropout_prob: float = DROPOUT_PROB,
        pretrained: bool = True,
        rnn_hidden_size: int = RNN_HIDDEN_SIZE,
        seq_len: int = SEQ_LEN,
        freeze_front: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = torchvision.models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        if freeze_front:
            for module in [self.cnn.conv1, self.cnn.bn1,
                           self.cnn.layer1, self.cnn.layer2]:
                for p in module.parameters():
                    p.requires_grad = False


        self.cnn.fc = nn.Identity()  
        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.rnn_hidden_size = rnn_hidden_size

        combined_dim = 512 + rnn_hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, output_size),
        )

    def forward(self, x: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        
        cnn_feat = self.cnn(x)  
        _, h_n = self.rnn(speed_seq)
        last_h = h_n[-1]  

        combined = torch.cat([cnn_feat, last_h], dim=1)  

        output = self.fc(combined) 

        return output
    