# Cell 1 Config
# ============================================================
#
# このセルの役割:
#
#   Cell 2からCell 11で使用する設定値を一括管理します。
#   通常の学習条件変更は、このセルだけを編集します。
#
# 設計ルール:
#
#   - 定数指定だけを置く
#   - モデル実装の固定値は各モデルクラスへ置く
#   - 関数、クラス、処理、例外処理は置かない
#   - 設定を縦に詰め、差分を確認しやすくする
#
# ============================================================


# Model
MODEL_NAME = "yolov5"
NUM_CLASSES = 1
CLASS_NAMES = {0: "face"}
DEVICE_NAME = "auto"
USE_PRETRAINED_WEIGHTS = True

# Dataset
IMAGE_SIZE = 640
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

# YOLOv5 Loss
YOLOV5_BOX_LOSS_WEIGHT = 0.05
YOLOV5_OBJECTNESS_LOSS_WEIGHT = 1.0
YOLOV5_CLASSIFICATION_LOSS_WEIGHT = 0.5
YOLOV5_OBJECTNESS_POSITIVE_WEIGHT = 1.0
YOLOV5_CLASSIFICATION_POSITIVE_WEIGHT = 1.0
YOLOV5_ANCHOR_MATCH_THRESHOLD = 4.0
YOLOV5_LABEL_SMOOTHING = 0.0
YOLOV5_FOCAL_GAMMA = 0.0
YOLOV5_FOCAL_ALPHA = 0.25
YOLOV5_OBJECTNESS_IOU_RATIO = 1.0
YOLOV5_OBJECTNESS_BALANCE = [4.0, 1.0, 0.4]

# Optimizer
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
SGD_MOMENTUM = 0.9

# Scheduler
SCHEDULER_NAME = "CosineAnnealingLR"
NUM_EPOCHS = 1
STEP_LR_STEP_SIZE = 10
STEP_LR_GAMMA = 0.1
COSINE_ANNEALING_MIN_LR = 1e-6

# Output
OUTPUT_DIRECTORY = "/kaggle/working/dlc26_outputs"
CHECKPOINT_DIRECTORY_NAME = "checkpoints"
CHECKPOINT_FILE_PREFIX = "epoch"
HISTORY_JSON_FILE_NAME = "training_history.json"
HISTORY_IMAGE_FILE_NAME = "training_history.png"
BEST_ONNX_FILE_NAME = "model_best.onnx"
FINAL_ONNX_FILE_NAME = "model_final.onnx"
SHOW_HISTORY = True

# ONNX
ONNX_OPSET_VERSION = 12
ONNX_DYNAMIC_BATCH = True
