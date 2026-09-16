# ============================================================
# Cell 2: Pretrained Faster R-CNN
# ============================================================

import torch
import torchvision

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device =", device)


weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

# 顔1クラス用に置換
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes=2
)

model.to(device)

print("Model Ready")