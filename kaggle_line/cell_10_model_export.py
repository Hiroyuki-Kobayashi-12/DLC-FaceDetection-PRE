# Cell 10 Model Export
# ============================================================
#
# このセルの役割:
#
#   学習チェックポイントの保存と読み込み、
#   学習済みモデルのONNX出力に必要な関数を作成します。
#
# 保存方針:
#
#   - 各エポックの学習結果を個別の.pthへ保存する
#   - Validation Lossが最小のモデルをbestとして管理する
#   - 最終エポックのモデルをfinalとして管理する
#   - ONNXはbestとfinalの2つだけ出力する
#
# Cell 10は保存部品だけを定義します。
# エポック番号、best判定、保存名、実行順はCell 11が管理します。
#
# Configから受け取る定数:
#
#   IMAGE_SIZE
#   ONNX_OPSET_VERSION
#   ONNX_DYNAMIC_BATCH
#
# 後続セルへ渡す関数:
#
#   save_checkpoint()
#   load_checkpoint_model()
#   export_onnx()
#
# ============================================================


from pathlib import Path
import copy

import torch


# ============================================================
# PyTorch Checkpoint Save
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    history,
    model_metadata,
    epoch,
    validation_loss,
    save_path,
):
    """1エポック分の学習状態を.pthへ保存します。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scheduler_state = (
        scheduler.state_dict()
        if scheduler is not None
        else None
    )

    checkpoint = {
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "history": history,
        "model_metadata": model_metadata,
        "image_size": IMAGE_SIZE,
    }

    torch.save(
        checkpoint,
        save_path,
    )


# ============================================================
# PyTorch Checkpoint Load
# ============================================================

def load_checkpoint_model(
    model,
    checkpoint_path,
):
    """.pthの重みをモデルへ読み込み、評価モードで返します。"""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    return model.cpu().eval()


# ============================================================
# YOLOv5 ONNX Export
# ============================================================

class YOLOv5Export(torch.nn.Module):
    """YOLOv5をONNX推論用モデルへ変換する専用クラスです。"""

    def __init__(self, model):
        """モデルを複製し、CPU上の評価モードへ移します。"""
        super().__init__()

        self.model = copy.deepcopy(model).cpu().eval()

    def yolov5_detection_setting(self):
        """YOLOv5 Detect HeadをONNX出力用に設定します。"""
        from models.yolo import Detect

        for module in self.model.modules():
            if isinstance(module, Detect):
                module.inplace = False
                module.dynamic = False
                module.export = False

    def yolov5_dynamic_axes(self):
        """Configに応じてバッチ次元だけを可変にします。"""
        if not ONNX_DYNAMIC_BATCH:
            return None

        return {
            "images": {
                0: "batch_size",
            },
            "predictions": {
                0: "batch_size",
            },
        }

    def forward(self, images):
        """YOLOv5のデコード済み推論出力だけを返します。"""
        return self.model(images)[0]

    def export(self, save_path):
        """YOLOv5をONNX形式で保存します。"""
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.yolov5_detection_setting()

        dummy_input = torch.zeros(
            (1, 3, IMAGE_SIZE, IMAGE_SIZE),
            dtype=torch.float32,
        )

        with torch.no_grad():
            self(dummy_input)

        torch.onnx.export(
            model=self,
            args=dummy_input,
            f=str(save_path),
            export_params=True,
            opset_version=ONNX_OPSET_VERSION,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["predictions"],
            dynamic_axes=self.yolov5_dynamic_axes(),
            dynamo=False,
        )


# ============================================================
# ONNX Export Select
# ============================================================

def export_onnx(
    model,
    model_metadata,
    save_path,
):
    """モデル系列に対応する専用クラスでONNXを出力します。"""
    if model_metadata["model_family"] == "yolov5":
        exporter = YOLOv5Export(
            model=model
        )

    exporter.export(
        save_path=save_path
    )
