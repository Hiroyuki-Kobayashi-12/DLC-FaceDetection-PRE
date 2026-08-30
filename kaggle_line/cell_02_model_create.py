# Cell 2 Model Create
# ============================================================
#
# このセルの役割:
#
#   Configで選択されたモデルを作成し、
#   後続セルで共通して扱える形式にそろえます。
#
# モデルごとに独立したハードクラスを作成し、
# モデル固有の構築処理は、対応するクラス内へ閉じ込めます。
#
# 現在実装しているモデル:
#
#   - YOLOv5
#
# 設計ルール:
#
#   - 共通親クラスを作らない
#   - 共通モデル生成関数を作らない
#   - モデル固有の処理をクラス外で共有しない
#   - クラス内の処理は小さな部品へ分割する
#   - create()の返り値形式だけを全モデルで統一する
#   - モデル固有の固定値は各モデルクラス内に置く
#   - 例外処理と確認用printは置かない
#
# Configから受け取る定数:
#
#   MODEL_NAME
#   NUM_CLASSES
#   CLASS_NAMES
#   DEVICE_NAME
#   USE_PRETRAINED_WEIGHTS
#
# 後続セルへ渡す変数:
#
#   model_result
#   model
#   DEVICE
#   MODEL_METADATA
#
# create()の統一返り値:
#
#   {
#       "model": model,
#       "device": device,
#       "model_name": model_name,
#       "model_family": model_family,
#       "num_classes": num_classes,
#       "class_names": class_names,
#   }
# ============================================================


from pathlib import Path
import subprocess
import sys

import torch


# ============================================================
# YOLOv5
# ============================================================

class YOLOv5:
    """従来型のアンカーありYOLOv5sを作成する専用クラスです。"""

    def __init__(self):
        """Configの共通設定とYOLOv5固有の固定設定を保持します。"""

        # Configから受け取るモデル共通設定です。
        self.selected_model_name = MODEL_NAME
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES
        self.device_name = DEVICE_NAME
        self.use_pretrained_weights = USE_PRETRAINED_WEIGHTS

        # YOLOv5固有設定です。
        # YOLOv5を変更するときだけ、このクラス内を編集します。
        self.model_name = "yolov5s"
        self.model_family = "yolov5"
        self.repository_version = "v7.0"
        self.repository_url = "https://github.com/ultralytics/yolov5.git"
        self.repository_directory = Path("/kaggle/working/yolov5")
        self.model_config_name = "yolov5s.yaml"
        self.pretrained_weights_name = "yolov5s.pt"

        # YOLOv5モデル設定ファイルのパスです。
        self.model_config_path = (
            self.repository_directory
            / "models"
            / self.model_config_name
        )

        # create()内で順番に作成する部品です。
        self.device = None
        self.model = None

    def yolov5_repository(self):
        """固定バージョンのYOLOv5リポジトリを準備します。"""
        if not self.repository_directory.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--branch",
                    self.repository_version,
                    "--depth",
                    "1",
                    self.repository_url,
                    str(self.repository_directory),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def yolov5_python_path(self):
        """YOLOv5リポジトリをNotebookからimport可能にします。"""
        repository_path = str(self.repository_directory)

        if repository_path not in sys.path:
            sys.path.insert(0, repository_path)

    def yolov5_device(self):
        """Configの共通指定からモデルを配置するdeviceを作ります。"""
        if self.device_name == "auto":
            selected_device_name = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            selected_device_name = self.device_name

        self.device = torch.device(selected_device_name)

    def yolov5_detection_model(self):
        """固定したYAMLから顔検出用YOLOv5sを作ります。"""
        from models.yolo import DetectionModel

        self.model = DetectionModel(
            cfg=str(self.model_config_path),
            ch=3,
            nc=self.num_classes,
            anchors=None,
        )

    def yolov5_pretrained_weights(self):
        """有効な場合だけ、構造が一致するCOCO事前学習重みを読み込みます。"""
        if not self.use_pretrained_weights:
            return

        from utils.downloads import attempt_download
        from utils.general import intersect_dicts

        weights_path = attempt_download(
            self.pretrained_weights_name
        )

        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=False,
        )

        pretrained_state_dict = (
            checkpoint["model"]
            .float()
            .state_dict()
        )

        compatible_state_dict = intersect_dicts(
            pretrained_state_dict,
            self.model.state_dict(),
            exclude=["anchor"],
        )

        self.model.load_state_dict(
            compatible_state_dict,
            strict=False,
        )

    def yolov5_detection_setting(self):
        """クラス情報を設定し、モデルを学習用deviceへ配置します。"""
        self.model.nc = self.num_classes
        self.model.names = self.class_names
        self.model = self.model.to(self.device)
        self.model.train()

    def yolov5_result(self):
        """YOLOv5の作成結果を全モデル共通の形式へまとめます。"""
        return {
            "model": self.model,
            "device": self.device,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }

    def create(self):
        """YOLOv5専用部品を順番に実行し、完成モデルを返します。"""
        self.yolov5_repository()
        self.yolov5_python_path()
        self.yolov5_device()
        self.yolov5_detection_model()
        self.yolov5_pretrained_weights()
        self.yolov5_detection_setting()

        return self.yolov5_result()


# ============================================================
# Model Select
# ============================================================
#
# ConfigのMODEL_NAMEで使用するハードクラスを選択します。
# モデルを追加するときは、独立したモデルクラスと選択分岐を追加します。
# 例外処理や共通Factoryは使用しません。
#
# ============================================================

if MODEL_NAME == "yolov5":
    detector = YOLOv5()


# ============================================================
# Model Create
# ============================================================

model_result = detector.create()


# ============================================================
# Output
# ============================================================

model = model_result["model"]
DEVICE = model_result["device"]

MODEL_METADATA = {
    "model_name": model_result["model_name"],
    "model_family": model_result["model_family"],
    "num_classes": model_result["num_classes"],
    "class_names": model_result["class_names"],
}
