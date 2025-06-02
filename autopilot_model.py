import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torch2trt import TRTModule
from torch2trt import torch2trt
from torchvision.models.feature_extraction import create_feature_extractor

OUTPUT_SIZE = 2
DROPOUT_PROB = 0.4

class AutopilotModel(torch.nn.Module):
    
    def __init__(self, pretrained:bool=True):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.network.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=DROPOUT_PROB),
            torch.nn.Linear(in_features=self.network.fc.in_features, out_features=128),
            torch.nn.Dropout(p=DROPOUT_PROB),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.Dropout(p=DROPOUT_PROB),
            torch.nn.Linear(in_features=64, out_features=OUTPUT_SIZE)
        )

        self.feature_extractor = create_feature_extractor(
            self.network,
            return_nodes = {"layer1": "feat1", "layer2": "feat2", "fc": "out"}
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        fx = self.feature_extractor(x)          # 한 번의 pass
        out   = fx["out"]                         # 최종 출력
        if return_feats and (not self.training):
            return out, fx["feat1"], fx["feat2"]                # (B,512,1,1)
        return out
    
    def save_to_path(self, path):
        torch.save(self.state_dict(), path)
        
    def load_from_path(self, path):
        self.load_state_dict(torch.load(path))
        
