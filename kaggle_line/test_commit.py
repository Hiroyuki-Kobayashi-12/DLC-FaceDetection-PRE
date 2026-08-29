# ============================================================
# Cell 1: WIDER FACE JSON Dataset Loader
# ============================================================
#
# このセルの目的:
#
#   1. Kaggle Dataset フォルダを自動検出する
#   2. train / val の annotation JSON を読み込む Dataset を作る
#   3. PyTorch DataLoader を作る
#   4. DEBUG=True のときだけ、bbox付き画像を可視化する
#
# このセルで作成される主な変数:
#
#   train_dataset
#   val_easy_dataset
#   val_medium_dataset
#   val_hard_dataset
#
#   train_loader
#   val_easy_loader
#   val_medium_loader
#   val_hard_loader
#
# 現在のDataset構造:
#
#   DATA_ROOT/
#   ├── train/
#   │   ├── image/
#   │   │   └── 0--Parade/xxx.jpg
#   │   └── anno/
#   │       └── 0--Parade/xxx.json
#   └── val/
#       ├── image/
#       │   └── 0--Parade/xxx.jpg
#       └── anno/
#           └── 0--Parade/xxx.json
#
# 現在のJSON構造:
#
#   {
#     "dataset": "WIDER FACE",
#     "split": "val",
#     "image": {
#       "file_name": "xxx.jpg",
#       "relative_path": "0--Parade/xxx.jpg",
#       "event": "0--Parade",
#       "width": 1024,
#       "height": 768
#     },
#     "faces": [
#       {
#         "face_index": 1,
#         "class_id": 0,
#         "class_name": "face",
#         "levels": ["easy", "medium", "hard"],
#         "bbox": {
#           "xyxy": [x1, y1, x2, y2],
#           "xywh": [x, y, w, h],
#           "yolo_xywh_norm": [xc, yc, w, h]
#         },
#         "valid_bbox": true,
#         "valid_for_training": true
#       }
#     ]
#   }
#
# 学習対象:
#
#   train:
#     face["valid_bbox"] == True
#     face["valid_for_training"] == True
#
#   val easy:
#     face["valid_bbox"] == True
#     "easy" in face["levels"]
#
#   val medium:
#     face["valid_bbox"] == True
#     "medium" in face["levels"]
#
#   val hard:
#     face["valid_bbox"] == True
#     "hard" in face["levels"]
#
# ============================================================


# ============================================================
# 1. ライブラリ読み込み
# ============================================================

from pathlib import Path
import json
import random

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ============================================================
# 2. 基本設定
# ============================================================

# DEBUG=True のとき、Datasetからランダムに画像を取り出してbboxを可視化します。
# 学習を早く進めたいときは False にしてください。
DEBUG = True

# DataLoader設定
BATCH_SIZE = 8
NUM_WORKERS = 2

# 顔検出なのでクラスは face の1クラスだけです。
CLASS_ID = 0
CLASS_NAME = "face"

# Kaggle Notebook では、追加したDatasetは通常 /kaggle/input 以下にあります。
KAGGLE_INPUT_DIR = Path("/kaggle/input")


# ============================================================
# 3. Dataset root 自動検出
# ============================================================

def find_data_root(input_dir: Path) -> Path:
    """
    /kaggle/input 以下から、以下の構造を持つフォルダを探します。

      train/image
      train/anno
      val/image
      val/anno

    Kaggle Dataset は、Dataset名や表示名によってパスが変わることがあります。
    そのため、固定パスではなく自動検出にしています。
    """
    for path in input_dir.rglob("*"):
        if not path.is_dir():
            continue

        train_image_dir = path / "train" / "image"
        train_anno_dir = path / "train" / "anno"
        val_image_dir = path / "val" / "image"
        val_anno_dir = path / "val" / "anno"

        if (
            train_image_dir.exists()
            and train_anno_dir.exists()
            and val_image_dir.exists()
            and val_anno_dir.exists()
        ):
            return path

    raise FileNotFoundError(
        "Dataset root not found. "
        "Expected train/image, train/anno, val/image, val/anno under /kaggle/input."
    )


DATA_ROOT = find_data_root(KAGGLE_INPUT_DIR)

TRAIN_IMAGE_DIR = DATA_ROOT / "train" / "image"
TRAIN_ANNO_DIR = DATA_ROOT / "train" / "anno"

VAL_IMAGE_DIR = DATA_ROOT / "val" / "image"
VAL_ANNO_DIR = DATA_ROOT / "val" / "anno"

print("========== Dataset Path ==========")
print(f"DATA_ROOT       : {DATA_ROOT}")
print(f"TRAIN_IMAGE_DIR : {TRAIN_IMAGE_DIR}")
print(f"TRAIN_ANNO_DIR  : {TRAIN_ANNO_DIR}")
print(f"VAL_IMAGE_DIR   : {VAL_IMAGE_DIR}")
print(f"VAL_ANNO_DIR    : {VAL_ANNO_DIR}")


# ============================================================
# 4. ファイル数確認
# ============================================================

def count_files(root: Path, suffix: str) -> int:
    """
    指定フォルダ以下のファイル数を数えます。

    suffix:
      ".jpg"
      ".json"
    """
    return len(list(root.rglob(f"*{suffix}")))


train_image_count = count_files(TRAIN_IMAGE_DIR, ".jpg")
train_json_count = count_files(TRAIN_ANNO_DIR, ".json")

val_image_count = count_files(VAL_IMAGE_DIR, ".jpg")
val_json_count = count_files(VAL_ANNO_DIR, ".json")

print("\n========== File Counts ==========")
print(f"train images : {train_image_count}")
print(f"train jsons  : {train_json_count}")
print(f"val images   : {val_image_count}")
print(f"val jsons    : {val_json_count}")

if train_image_count != train_json_count:
    print("[WARN] train image count and json count are different.")

if val_image_count != val_json_count:
    print("[WARN] val image count and json count are different.")


# ============================================================
# 5. JSON / 画像変換ユーティリティ
# ============================================================

def load_json(path: Path) -> dict:
    """
    JSON annotation を読み込みます。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    PIL画像を PyTorch Tensor に変換します。

    入力:
      PIL RGB image

    出力:
      torch.FloatTensor
      shape: [3, H, W]
      range: 0.0 - 1.0
    """
    array = np.array(image, dtype=np.float32) / 255.0

    # [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    return tensor


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    PyTorch Tensor を PIL画像に戻します。

    入力:
      shape: [3, H, W]
      range: 0.0 - 1.0

    出力:
      PIL RGB image
    """
    array = image_tensor.permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(array)


# ============================================================
# 6. bbox処理
# ============================================================

def clip_xyxy(box, width: int, height: int):
    """
    bboxを画像サイズ内に収めます。

    入力:
      box = [x1, y1, x2, y2]

    出力:
      [x1, y1, x2, y2]

    bboxが壊れている場合は None を返します。
    """
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def face_is_target(face: dict, split: str, level: str) -> bool:
    """
    顔bboxを使うかどうか判定します。

    train:
      valid_bbox=True かつ valid_for_training=True の顔を使います。

    val:
      level="easy":
        levels に "easy" を含む顔を使います。

      level="medium":
        levels に "medium" を含む顔を使います。

      level="hard":
        levels に "hard" を含む顔を使います。

      level="all":
        valid_bbox=True の顔を使います。
    """
    if not bool(face.get("valid_bbox", False)):
        return False

    if split == "train":
        return bool(face.get("valid_for_training", False))

    if level == "all":
        return True

    return level in face.get("levels", [])


def extract_target_from_anno(anno: dict, split: str, level: str):
    """
    JSON annotation から PyTorch detection 用 target を作ります。

    戻り値:
      boxes:
        FloatTensor[N, 4]
        xyxy形式

      labels:
        LongTensor[N]
        全て class_id=0
    """
    width = int(anno["image"]["width"])
    height = int(anno["image"]["height"])

    boxes = []
    labels = []

    for face in anno["faces"]:
        if not face_is_target(face, split=split, level=level):
            continue

        box = face["bbox"]["xyxy"]

        if box is None:
            continue

        box = clip_xyxy(box, width=width, height=height)

        if box is None:
            continue

        boxes.append(box)
        labels.append(int(face["class_id"]))

    if len(boxes) == 0:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.long)
    else:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

    return boxes_tensor, labels_tensor


# ============================================================
# 7. PyTorch Dataset
# ============================================================

class FaceJsonDataset(Dataset):
    """
    顔検出用 JSON Dataset。

    Parameters
    ----------
    split:
      "train" または "val"

    level:
      val の難易度指定です。

      "all"
      "easy"
      "medium"
      "hard"

      trainでは実質使いません。

    drop_empty:
      True の場合、指定条件でbboxが0個の画像を除外します。

      val_easy / val_medium / val_hard では True にします。
      これにより、可視化や評価時にbboxが空の画像を避けやすくなります。
    """

    def __init__(
        self,
        split: str,
        level: str = "all",
        drop_empty: bool = False,
    ):
        if split not in ["train", "val"]:
            raise ValueError("split must be 'train' or 'val'")

        if level not in ["all", "easy", "medium", "hard"]:
            raise ValueError("level must be one of: all, easy, medium, hard")

        self.split = split
        self.level = level
        self.drop_empty = drop_empty

        if split == "train":
            self.image_dir = TRAIN_IMAGE_DIR
            self.anno_dir = TRAIN_ANNO_DIR
        else:
            self.image_dir = VAL_IMAGE_DIR
            self.anno_dir = VAL_ANNO_DIR

        self.anno_files = sorted(self.anno_dir.rglob("*.json"))

        if len(self.anno_files) == 0:
            raise FileNotFoundError(f"No annotation JSON files found: {self.anno_dir}")

        if self.drop_empty:
            self.anno_files = self._filter_non_empty(self.anno_files)

    def _filter_non_empty(self, anno_files):
        """
        指定条件でbboxが1つ以上あるJSONだけ残します。
        """
        filtered = []

        for anno_path in anno_files:
            anno = load_json(anno_path)

            boxes, _ = extract_target_from_anno(
                anno=anno,
                split=self.split,
                level=self.level,
            )

            if boxes.shape[0] > 0:
                filtered.append(anno_path)

        return filtered

    def __len__(self):
        return len(self.anno_files)

    def __getitem__(self, index):
        anno_path = self.anno_files[index]
        anno = load_json(anno_path)

        relative_path = anno["image"]["relative_path"]
        image_path = self.image_dir / relative_path

        image = Image.open(image_path).convert("RGB")
        image_tensor = pil_to_tensor(image)

        boxes, labels = extract_target_from_anno(
            anno=anno,
            split=self.split,
            level=self.level,
        )

        target = {
            "boxes": boxes,
            "labels": labels,

            "image_path": str(image_path),
            "anno_path": str(anno_path),
            "relative_path": relative_path,

            "width": int(anno["image"]["width"]),
            "height": int(anno["image"]["height"]),

            "split": self.split,
            "level": self.level,

            # 勉強会でJSONの中身を確認しやすいように残します。
            "raw_anno": anno,
        }

        return image_tensor, target


def collate_fn(batch):
    """
    物体検出では、画像ごとにbbox数が異なります。

    通常のDataLoaderのようにTensorへstackせず、
    images, targets を list のまま返します。
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


# ============================================================
# 8. Dataset作成
# ============================================================

train_dataset = FaceJsonDataset(
    split="train",
    level="all",
    drop_empty=False,
)

# val_all_dataset は、val全体の顔数確認用です。
# DataLoaderとして必ず使うわけではありませんが、
# easy / medium / hard 以外に「val全体で有効bboxが何個あるか」を確認できます。
val_all_dataset = FaceJsonDataset(
    split="val",
    level="all",
    drop_empty=False,
)

val_easy_dataset = FaceJsonDataset(
    split="val",
    level="easy",
    drop_empty=True,
)

val_medium_dataset = FaceJsonDataset(
    split="val",
    level="medium",
    drop_empty=True,
)

val_hard_dataset = FaceJsonDataset(
    split="val",
    level="hard",
    drop_empty=True,
)


# ============================================================
# 8.1 顔矩形数カウント
# ============================================================

def count_raw_faces(dataset: FaceJsonDataset) -> int:
    """
    Dataset内のJSONを見て、元の faces 配列に入っている顔数を数えます。

    これは valid_bbox や valid_for_training に関係なく、
    JSONに入っている顔アノテーション総数です。
    """
    total_faces = 0

    for anno_path in dataset.anno_files:
        anno = load_json(anno_path)
        total_faces += len(anno["faces"])

    return total_faces


def count_target_faces(dataset: FaceJsonDataset) -> int:
    """
    Datasetの split / level 条件で実際に使われる顔bbox数を数えます。

    train_dataset:
      valid_bbox=True かつ valid_for_training=True の顔数

    val_all_dataset:
      valid_bbox=True の顔数

    val_easy_dataset:
      valid_bbox=True かつ "easy" in levels の顔数

    val_medium_dataset:
      valid_bbox=True かつ "medium" in levels の顔数

    val_hard_dataset:
      valid_bbox=True かつ "hard" in levels の顔数

    注意:
      val の easy / medium / hard は累積的に数えます。
      つまり levels=["easy", "medium", "hard"] の顔は、
      easy, medium, hard すべてで1つずつカウントされます。
    """
    total_faces = 0

    for anno_path in dataset.anno_files:
        anno = load_json(anno_path)

        boxes, _ = extract_target_from_anno(
            anno=anno,
            split=dataset.split,
            level=dataset.level,
        )

        total_faces += int(boxes.shape[0])

    return total_faces


def count_val_levels_from_json(dataset: FaceJsonDataset) -> dict:
    """
    val JSONの faces 配列を直接見て、
    easy / medium / hard のlevel数を集計します。

    1つの顔が複数levelを持っている場合は、全てカウントします。

    例:
      levels = ["easy", "medium", "hard"]

    この場合:
      easy   +1
      medium +1
      hard   +1
    """
    counts = {
        "easy": 0,
        "medium": 0,
        "hard": 0,
        "unknown": 0,
    }

    for anno_path in dataset.anno_files:
        anno = load_json(anno_path)

        for face in anno["faces"]:
            if not bool(face.get("valid_bbox", False)):
                continue

            levels = face.get("levels", [])

            if len(levels) == 0:
                counts["unknown"] += 1
                continue

            if "easy" in levels:
                counts["easy"] += 1

            if "medium" in levels:
                counts["medium"] += 1

            if "hard" in levels:
                counts["hard"] += 1

    return counts


train_raw_faces = count_raw_faces(train_dataset)
val_raw_faces = count_raw_faces(val_all_dataset)

train_target_faces = count_target_faces(train_dataset)
val_all_faces = count_target_faces(val_all_dataset)

val_easy_faces = count_target_faces(val_easy_dataset)
val_medium_faces = count_target_faces(val_medium_dataset)
val_hard_faces = count_target_faces(val_hard_dataset)

val_level_counts = count_val_levels_from_json(val_all_dataset)


print("\n========== Dataset Length ==========")
print(f"train_dataset     : {len(train_dataset)}")
print(f"val_all_dataset   : {len(val_all_dataset)}")
print(f"val_easy_dataset  : {len(val_easy_dataset)}")
print(f"val_medium_dataset: {len(val_medium_dataset)}")
print(f"val_hard_dataset  : {len(val_hard_dataset)}")

print("\n========== Face Box Counts ==========")
print(f"train raw faces          : {train_raw_faces}")
print(f"train target faces       : {train_target_faces}")
print(f"val raw faces            : {val_raw_faces}")
print(f"val all valid faces      : {val_all_faces}")

print("\n========== Val Level Face Counts ==========")
print(f"val easy faces           : {val_easy_faces}")
print(f"val medium faces         : {val_medium_faces}")
print(f"val hard faces           : {val_hard_faces}")

print("\n========== Val Level Counts From JSON ==========")
print(f"easy                     : {val_level_counts['easy']}")
print(f"medium                   : {val_level_counts['medium']}")
print(f"hard                     : {val_level_counts['hard']}")
print(f"unknown                  : {val_level_counts['unknown']}")


# ============================================================
# 9. DataLoader作成
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
)

val_easy_loader = DataLoader(
    val_easy_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
)

val_medium_loader = DataLoader(
    val_medium_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
)

val_hard_loader = DataLoader(
    val_hard_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
)


# ============================================================
# 10. 1バッチ確認
# ============================================================

images, targets = next(iter(train_loader))

print("\n========== Batch Check ==========")
print(f"batch size         : {len(images)}")
print(f"first image shape  : {images[0].shape}")
print(f"first image range  : {images[0].min().item():.3f} - {images[0].max().item():.3f}")
print(f"first boxes shape  : {targets[0]['boxes'].shape}")
print(f"first labels shape : {targets[0]['labels'].shape}")
print(f"first path         : {targets[0]['relative_path']}")


# ============================================================
# 11. 可視化関数
# ============================================================

def draw_boxes(
    image_tensor: torch.Tensor,
    target: dict,
    color=(255, 0, 0),
) -> Image.Image:
    """
    Tensor画像にbboxを描画して、PIL画像として返します。
    """
    image = tensor_to_pil(image_tensor)
    draw = ImageDraw.Draw(image)

    for box in target["boxes"]:
        x1, y1, x2, y2 = box.tolist()

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=3,
        )

    return image


def show_random_samples(
    dataset,
    title: str,
    n: int,
    color=(255, 0, 0),
    cols: int = 4,
):
    """
    Datasetからランダムに n 枚選んで表示します。

    dataset が空の場合は、エラーにせずメッセージだけ表示します。
    """
    if len(dataset) == 0:
        print(f"[WARN] {title}: dataset is empty. Skip visualization.")
        return

    n = min(n, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 4, rows * 4))

    for plot_index, data_index in enumerate(indices):
        image_tensor, target = dataset[data_index]
        image = draw_boxes(image_tensor, target, color=color)

        plt.subplot(rows, cols, plot_index + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"{target['split']} / {target['level']} / boxes={len(target['boxes'])}",
            fontsize=10,
        )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================
# 12. DEBUG可視化
# ============================================================

if DEBUG:
    print("\n========== DEBUG Visualization ==========")

    # train: ランダム16枚
    show_random_samples(
        dataset=train_dataset,
        title="Train samples",
        n=16,
        color=(0, 200, 255),
        cols=4,
    )

    # val: easy / medium / hard をそれぞれ4枚ずつ
    show_random_samples(
        dataset=val_easy_dataset,
        title="Val easy samples",
        n=4,
        color=(0, 220, 0),
        cols=4,
    )

    show_random_samples(
        dataset=val_medium_dataset,
        title="Val medium samples",
        n=4,
        color=(255, 180, 0),
        cols=4,
    )

    show_random_samples(
        dataset=val_hard_dataset,
        title="Val hard samples",
        n=4,
        color=(255, 0, 0),
        cols=4,
    )


# ============================================================
# 13. 完了
# ============================================================

print("\n========== Cell 1 Ready ==========")
print("DataLoaders are ready:")
print("  train_loader")
print("  val_easy_loader")
print("  val_medium_loader")
print("  val_hard_loader")