# ============================================================
# Cell 2: YOLOv1スタイル 顔検出モデル — 学習環境の実装
# ============================================================
#
# 前提:
#   Cell 1 で作成済みの以下の変数が、このセルより前に実行されていること。
#
#     train_dataset
#     val_easy_dataset / val_medium_dataset / val_hard_dataset
#
#   (train_loader / val_*_loader は Cell 1 のものを使わず、
#    このセルで新しい collate_fn を使って作り直します。
#    理由は下記 1. で説明します)
#
# Kaggle Notebook で実行する際の前提・注意点:
#
#   - Notebook設定の Accelerator を GPU (T4 x2 など) にしてください。
#     CPUのままでは学習に非常に時間がかかります。
#
#   - Notebook設定の Internet を ON にしてください。
#     モデル定義 (下記 2.) で ResNet34 の ImageNet 事前学習済み重みを
#     ダウンロードするため、インターネット接続が必要です。
#
#   - このセルは Python スクリプト (.py) ではなく Notebook のセルとして
#     実行される前提のため、スクリプト向けの `if __name__ == "__main__":`
#     は使わず、学習ループ (下記 8.) はセル内で直接実行されるようにしています。
#     その代わり、下記の RUN_TRAINING スイッチで
#     「まず定義と1バッチ動作確認だけ行う」か「学習まで一気に回す」かを
#     切り替えられるようにしています。
#
# このセルの目的:
#
#   1. 教師データの符号化 (Target Encoding)
#      xyxy形式のbboxリスト → 7×7グリッドの教師テンソルへの変換
#
#   2. モデル本体の実装
#      backbone (ResNet34, 事前学習済み) + head (7×7×(B*5+C) 出力)
#
#   3. 損失関数の実装 (YOLOv1の5項の重み付き二乗和)
#
#   4. 学習ループの実装
#
#   4.5 学習過程の可視化 (YOLO特有の3工程を目で確認する)
#      ・学習開始前   : ① グリッド分割と担当セルの確認 (符号化のバグ検出)
#      ・エポック終了時: ② 責任box/非責任boxとIoU値
#                        ③ confidence・クラス確率のヒートマップと数値
#      毎バッチではなく上記タイミングに絞ることで、学習速度への影響を抑えています。
#
# 設計上の主な決定と理由:
#
#   - 画像サイズ: 448×448 に統一 (YOLOv1論文と同じ)
#   - リサイズ方式: 縦横比を保たない単純な引き伸ばし
#     (以前解説した通りYOLOv1の標準的なやり方。実装が単純なため採用)
#   - グリッド: S=7, 1セルあたりの予測box数: B=2
#   - クラス数: C=1 (face のみ。WIDER FACEは顔検出のみのデータセットのため)
#   - 「どのboxが責任を持つか(responsible box)」の判定は、
#     学習中の予測とGTのIoUに基づいて動的に決まるため、
#     教師データの符号化時点では確定させず、損失関数の内部で毎回計算します。
#     (v1論文の定義に忠実な実装です)
#
# ============================================================


# ============================================================
# 0. ライブラリ読み込みと基本設定
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

# ---- ハイパーパラメータ (人間が決める設定値) ----
IMG_SIZE = 448      # ネットワークへの入力画像サイズ (正方形)
S = 7                # グリッドの分割数 (S×S マス)
B = 2                # 1セルあたりの予測box数
C = 1                # クラス数 (face の1クラスのみ)

LAMBDA_COORD = 5.0   # 座標損失の重み (以前解説した λcoord)
LAMBDA_NOOBJ = 0.5   # 物体なしconfidence損失の重み (以前解説した λnoobj)

BATCH_SIZE = 8       # Cell 1 と揃えています
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3      # まずは動作確認のための少ない値。後で調整してください

# True にすると、このセルの最後まで実行したときに本格的な学習ループ (手順8) まで
# 一気に走ります。まずはモデルの形やエラーの有無だけ確認したい場合は False にしてください。
# (False の場合、手順7の1バッチ動作確認までで処理が止まります)
RUN_TRAINING = True

# ---- デバイス選択 ----
# GPUが使えればGPUを、無ければCPUを使います。
# 注意: このモデルをCPU (特にMacBook Proのような非GPU環境) で学習すると
#       非常に時間がかかります。Kaggle Notebookの GPU設定 (T4など) を
#       有効にして実行することを強く推奨します。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- GPUの互換性チェック ----
# Kaggle Notebookで選べる GPU の中には、Compute Capability が低く
# (例: P100 = 6.0)、現在インストールされているPyTorchのビルドが
# 対応する範囲 (通常 7.0以降) の外にあるものがあります。
# この場合、モデルの計算を実行した瞬間に
# 「CUDA error: no kernel image is available for execution on the device」
# という分かりにくいエラーになるため、ここで先に気づけるようにチェックします。
if DEVICE.type == "cuda":
    capability_major, capability_minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)

    if capability_major < 7:
        raise RuntimeError(
            f"GPU '{gpu_name}' (Compute Capability {capability_major}.{capability_minor}) は、"
            "現在インストールされているPyTorchのビルドではサポートされていません "
            "(対応範囲は Compute Capability 7.0 以降です)。\n"
            "\n"
            "【対処方法】\n"
            "Kaggle Notebook右側パネルの [Settings] → [Accelerator] を、\n"
            "'GPU P100' から 'GPU T4 x2' に変更し、Notebookのセッションを\n"
            "再起動 (Restart session) してから、このセルを実行し直してください。"
        )

print(f"使用デバイス: {DEVICE}  (GPU: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'なし'})")


# ============================================================
# 1. 教師データの符号化 (Target Encoding)
# ============================================================
#
# Cell 1 の Dataset がそのまま返す boxes は、
#   ・元画像サイズのまま (リサイズされていない)
#   ・xyxy形式 [x1, y1, x2, y2] のリスト
# です。これを次の2段階で、モデルが学習に使えるテンソルに変換します。
#
#   (a) resize_image_and_boxes : 画像とbboxを448×448基準にリサイズ
#   (b) encode_target          : リサイズ後のbboxを7×7グリッドの
#                                 教師テンソル (S, S, 5+C) に変換
#

def resize_image_and_boxes(image_tensor: torch.Tensor, boxes_xyxy: torch.Tensor, size: int = IMG_SIZE):
    """
    画像とbboxを、縦横比を保たない単純な引き伸ばしで size×size にリサイズします。

    入力:
      image_tensor : [3, H, W]  (Cell 1 の pil_to_tensor が返す形式、0.0-1.0)
      boxes_xyxy   : [N, 4]     (元画像サイズでのピクセル座標)

    出力:
      resized_image : [3, size, size]
      resized_boxes : [N, 4]   (size×size 基準のピクセル座標)
    """
    _, height, width = image_tensor.shape

    # 画像のリサイズ (bilinear = 一般的な滑らかな補間方法)
    # interpolateはバッチ次元を要求するため、unsqueeze/squeezeで次元を調整します。
    resized_image = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # bboxのリサイズ: x方向とy方向で別々の倍率をかけます。
    # (これが「縦横比を保たない」ことに相当します)
    scale_x = size / width
    scale_y = size / height

    if boxes_xyxy.shape[0] == 0:
        resized_boxes = boxes_xyxy
    else:
        resized_boxes = boxes_xyxy.clone()
        resized_boxes[:, [0, 2]] *= scale_x   # x1, x2
        resized_boxes[:, [1, 3]] *= scale_y   # y1, y2

    return resized_image, resized_boxes


def encode_target(boxes_xyxy: torch.Tensor, labels: torch.Tensor, img_size: int = IMG_SIZE, s: int = S, c: int = C):
    """
    リサイズ後のbboxを、7×7グリッドの教師テンソルに変換します。

    出力テンソルの形状: (S, S, 5+C)
      [..., 0:2] = x_cell, y_cell  (担当セル内でのオフセット、0〜1)
      [..., 2:4] = w_norm, h_norm  (画像全体に対する比率、0〜1)
      [..., 4]   = confidence の目標値 (物体があれば1、なければ0)
      [..., 5:]  = クラスの one-hot (C=1 なので実質 [1.0] か [0.0])

    ルール:
      GT boxの中心が入るセルが、その物体の予測を担当する
      (以前解説した v1 の役割分担ルール)。

      1セルに複数のGT中心が入った場合、v1の制約通り
      「先に処理した1つだけ」を採用し、以降は無視します
      (1セル1物体という v1 の既知の弱点を、そのままコードにも反映しています)。
    """
    target = torch.zeros((s, s, 5 + c), dtype=torch.float32)
    cell_size = img_size / s

    for box, label in zip(boxes_xyxy, labels):
        x1, y1, x2, y2 = box.tolist()

        # 中心座標と幅・高さ (ピクセル単位)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        # 面積が0以下のbboxは無効なのでスキップします
        # (Cell 1 の clip_xyxy を通過していても、リサイズ誤差などで
        #  ごく稀に幅・高さが0近くになるケースへの防御的な処理です)
        if w <= 0 or h <= 0:
            continue

        # 担当セルの行(i)・列(j)インデックスを求めます
        j = int(cx // cell_size)   # 列方向 (x)
        i = int(cy // cell_size)   # 行方向 (y)

        # 中心が画像の端ギリギリの場合、計算上 S を超えることがあるため
        # 安全のため範囲内にクリップします
        i = min(max(i, 0), s - 1)
        j = min(max(j, 0), s - 1)

        # このセルにすでに別の物体が割り当て済みなら、v1の制約通りスキップします
        if target[i, j, 4] == 1.0:
            continue

        # セル内でのオフセット (0〜1に正規化)
        x_cell = cx / cell_size - j
        y_cell = cy / cell_size - i

        # 画像全体に対する幅・高さの比率
        w_norm = w / img_size
        h_norm = h / img_size

        target[i, j, 0] = x_cell
        target[i, j, 1] = y_cell
        target[i, j, 2] = w_norm
        target[i, j, 3] = h_norm
        target[i, j, 4] = 1.0
        target[i, j, 5 + int(label.item())] = 1.0

    return target


def collate_fn_yolo(batch):
    """
    Cell 1 の collate_fn とは別に、YOLO学習専用に用意する collate_fn です。

    Cell 1 のDatasetが返す (image_tensor, target_dict) のペアのリストを受け取り、
    リサイズ・教師データ符号化まで行った上で、バッチとしてまとめて返します。

    出力:
      images  : [N, 3, IMG_SIZE, IMG_SIZE]  (通常のTensor、stack済み)
      targets : [N, S, S, 5+C]              (通常のTensor、stack済み)

    Cell 1 の collate_fn がリストのまま返していたのに対し、
    ここでは全画像を同じサイズにリサイズしているため、
    通常通り torch.stack でまとめられる点が異なります。
    """
    images = []
    targets = []

    for image_tensor, target_dict in batch:
        resized_image, resized_boxes = resize_image_and_boxes(
            image_tensor, target_dict["boxes"], size=IMG_SIZE
        )
        target_tensor = encode_target(
            resized_boxes, target_dict["labels"], img_size=IMG_SIZE, s=S, c=C
        )

        images.append(resized_image)
        targets.append(target_tensor)

    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)

    return images, targets


# ============================================================
# 2. モデル本体の実装
# ============================================================
#
# 構成: backbone (ResNet34、事前学習済み) + head (検出用の畳み込み層)
#
# backboneをゼロから自作せず事前学習済みモデルを使う理由:
#   ・ImageNetで学習済みの重みを初期値にすることで、少ないデータ・
#     短い学習時間でも安定して収束しやすくなります (転移学習)
#   ・自作の24層CNN (YOLOv1論文オリジナル) より実装ミスのリスクが低く、
#     再現性の高い構成になります
#

class SimpleYoloFace(nn.Module):
    def __init__(self, s: int = S, b: int = B, c: int = C, pretrained: bool = True):
        super().__init__()
        self.s = s
        self.b = b
        self.c = c

        # ---- backbone ----
        # ResNet34の「畳み込み部分だけ」を取り出して使います。
        # list(backbone.children())[:-2] で、最後の
        # AdaptiveAvgPool2d と全結合層(分類用の部分)を除外しています。
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet34(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone_out_channels = 512   # ResNet34の最終畳み込み層の出力チャンネル数

        # ---- head ----
        # backboneの出力する特徴マップの解像度は入力サイズに依存して変わりますが、
        # AdaptiveAvgPool2d((S, S)) を挟むことで、解像度によらず必ず
        # S×S のグリッドに変換されるようにしています。
        self.head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((s, s)),
            nn.Conv2d(512, b * 5 + c, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: x [N, 3, IMG_SIZE, IMG_SIZE]
        出力: [N, S, S, B*5+C]
        """
        features = self.backbone(x)          # [N, 512, H', W']
        out = self.head(features)            # [N, B*5+C, S, S]
        out = out.permute(0, 2, 3, 1)         # [N, S, S, B*5+C]  空間軸を前に、チャンネルを最後に
        out = out.contiguous()
        return out


# ============================================================
# 3. 損失関数の実装 (YOLOv1の5項の重み付き二乗和)
# ============================================================

class YoloV1Loss(nn.Module):
    def __init__(self, s: int = S, b: int = B, c: int = C,
                 lambda_coord: float = LAMBDA_COORD, lambda_noobj: float = LAMBDA_NOOBJ):
        super().__init__()
        self.s = s
        self.b = b
        self.c = c
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

        # 各セルの左上座標 (セル単位、0〜S-1) をあらかじめ計算しておきます。
        # box座標をIoU計算のためxyxy形式に戻す際に使います。
        grid_y, grid_x = torch.meshgrid(
            torch.arange(s, dtype=torch.float32),
            torch.arange(s, dtype=torch.float32),
            indexing="ij",
        )
        # register_buffer: 学習対象のパラメータではないが、
        # model.to(device) と一緒にGPUへ転送したい定数として登録します。
        self.register_buffer("grid_x", grid_x)   # [S, S]
        self.register_buffer("grid_y", grid_y)   # [S, S]

    def _cellbox_to_corners(self, box: torch.Tensor) -> torch.Tensor:
        """
        セル基準の box [x_cell, y_cell, w_norm, h_norm] を、
        画像全体基準 (0〜1) の xyxy 座標に変換します。
        IoU計算のための前処理です。

        入力: box [..., S, S, 4] または [..., S, S, B, 4]
        出力: 同じ形の最後の次元が4のxyxyテンソル
        """
        cell_size = 1.0 / self.s

        # grid_x, grid_yの形状をboxにブロードキャストできるようにします
        gx = self.grid_x
        gy = self.grid_y
        while gx.dim() < box.dim() - 1:
            gx = gx.unsqueeze(0)
            gy = gy.unsqueeze(0)

        cx = (gx + box[..., 0]) * cell_size
        cy = (gy + box[..., 1]) * cell_size
        w = box[..., 2]
        h = box[..., 3]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        2つのxyxy形式のboxテンソル(同じ形)から、要素ごとのIoUを計算します。
        """
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])

        intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
        area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
        union = area1 + area2 - intersection

        return intersection / union.clamp(min=1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred   : [N, S, S, B*5+C]  (モデルの出力)
        target : [N, S, S, 5+C]    (encode_target で作った教師データ)

        戻り値:
          total_loss : スカラー (このバッチの平均損失)
          logs       : 内訳を確認するための辞書 (学習ログ表示用)
        """
        n = pred.shape[0]

        obj_mask = target[..., 4] == 1.0     # [N,S,S]  物体ありセルのbool マスク

        # ---- predをbox予測とクラス予測に分解 ----
        pred_boxes = pred[..., : self.b * 5].reshape(n, self.s, self.s, self.b, 5)
        pred_class = pred[..., self.b * 5:]                     # [N,S,S,C]

        target_box = target[..., 0:4]     # [N,S,S,4]
        target_class = target[..., 5:]    # [N,S,S,C]

        # ---- 責任box(responsible box)の決定 ----
        # 2つの予測boxそれぞれとGTのIoUを計算し、IoUが高い方を「責任box」とします。
        target_corners = self._cellbox_to_corners(target_box)   # [N,S,S,4]

        ious = []
        for box_index in range(self.b):
            pred_corners = self._cellbox_to_corners(pred_boxes[..., box_index, 0:4])
            ious.append(self._iou(pred_corners, target_corners))
        ious = torch.stack(ious, dim=-1)          # [N,S,S,B]

        best_box = torch.argmax(ious, dim=-1)                     # [N,S,S] 値は0またはB-1まで
        best_box_onehot = F.one_hot(best_box, num_classes=self.b).float()  # [N,S,S,B]

        obj_mask_f = obj_mask.float().unsqueeze(-1)                # [N,S,S,1]
        resp_mask = best_box_onehot * obj_mask_f                   # [N,S,S,B] 責任boxのみ1

        # ---- (1)(2) 座標損失 ----
        pred_xy = pred_boxes[..., 0:2]                              # [N,S,S,B,2]
        pred_wh = pred_boxes[..., 2:4]                              # [N,S,S,B,2]
        target_xy = target_box[..., 0:2].unsqueeze(3)               # [N,S,S,1,2] → broadcast
        target_wh = target_box[..., 2:4].unsqueeze(3)

        resp_mask_xy = resp_mask.unsqueeze(-1)                      # [N,S,S,B,1]

        xy_loss = self.mse(resp_mask_xy * pred_xy, resp_mask_xy * target_xy)

        # 幅・高さは平方根で比較します (以前解説した「小さいboxのずれを重く扱う」ための工夫)。
        # 学習初期は予測が負の値になることもあるため、符号を保ったまま絶対値の平方根をとります。
        pred_wh_sqrt = torch.sign(pred_wh) * torch.sqrt(pred_wh.abs().clamp(min=1e-6))
        target_wh_sqrt = torch.sqrt(target_wh.clamp(min=1e-6))
        wh_loss = self.mse(resp_mask_xy * pred_wh_sqrt, resp_mask_xy * target_wh_sqrt)

        coord_loss = self.lambda_coord * (xy_loss + wh_loss)

        # ---- (3) confidence損失 (物体あり、責任boxのみ) ----
        # 目標値は「予測box自身とGTのIoU」です (v1論文の定義通り)。
        pred_conf = pred_boxes[..., 4]                              # [N,S,S,B]
        target_conf_resp = ious.detach() * resp_mask                # 責任boxの位置だけIoU値、他は0
        obj_conf_loss = self.mse(resp_mask * pred_conf, target_conf_resp)

        # ---- (4) confidence損失 (物体なし) ----
        # 「物体のないセルの全box」と「物体はあるが責任でない方のbox」の両方を対象にします。
        noobj_mask_cell = (~obj_mask).float().unsqueeze(-1).expand(-1, -1, -1, self.b)  # [N,S,S,B]
        non_resp_mask = (1.0 - best_box_onehot) * obj_mask_f        # 物体ありセルの非責任box
        noobj_total_mask = noobj_mask_cell + non_resp_mask

        noobj_loss = self.lambda_noobj * self.mse(
            noobj_total_mask * pred_conf, torch.zeros_like(pred_conf)
        )

        # ---- (5) クラス確率損失 (物体のあるセルのみ) ----
        class_loss = self.mse(obj_mask_f * pred_class, obj_mask_f * target_class)

        total_loss = (coord_loss + obj_conf_loss + noobj_loss + class_loss) / n

        logs = {
            "total": total_loss.item(),
            "coord": coord_loss.item() / n,
            "obj": obj_conf_loss.item() / n,
            "noobj": noobj_loss.item() / n,
            "class": class_loss.item() / n,
        }

        return total_loss, logs


# ============================================================
# 4. 学習ループの実装
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    1エポック分、学習データ全体を1周して学習します。
    戻り値はこのエポックの平均損失です。
    """
    model.train()
    running_loss = 0.0

    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # ---- 順伝播 ----
        preds = model(images)
        loss, logs = loss_fn(preds, targets)

        # ---- 逆伝播とパラメータ更新 ----
        optimizer.zero_grad()
        loss.backward()

        # 勾配クリッピング: 勾配が異常に大きくなって学習が発散するのを防ぐ安全策です。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        running_loss += loss.item()

        if batch_index % 50 == 0:
            print(
                f"  batch {batch_index:4d}/{len(loader)}  "
                f"loss={logs['total']:.3f}  "
                f"(coord={logs['coord']:.3f} obj={logs['obj']:.3f} "
                f"noobj={logs['noobj']:.3f} class={logs['class']:.3f})"
            )

    return running_loss / len(loader)


# ============================================================
# 4.5 学習過程を確認するための可視化
# ============================================================
#
# YOLO特有の3つの工程が、実際にどのような処理として動いているかを
# 学習の進行に合わせて確認するための関数群です。
#
#   ① グリッド分割    : 画像を S×S に分け、GTの中心が入るセルが担当になる
#   ② bbox推定        : 担当セルの B 個の予測boxのうち、GTとのIoUが高い方が
#                        「責任box」として損失を受け持つ
#   ③ クラス予測      : confidence とクラス確率で「顔らしさ」を予測する
#
# 使い方の設計:
#   ・学習開始前に1回  → 教師データの符号化が正しくできているかの確認
#   ・各エポック終了時 → 学習が進むにつれて予測がどう変化するかの追跡
#
#   毎バッチ実行すると学習が極端に遅くなるため、上記のタイミングに絞っています。
#   また毎回同じ画像 (VIS_SAMPLE_INDEX) を使うことで、エポック間の変化を
#   比較しやすくしています。

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ---- matplotlibで日本語を文字化けさせないための設定 ----
# Kaggle Notebookの標準環境には日本語フォントが入っていないことが多く、
# そのままだとグラフのタイトルが四角い記号 (いわゆる「豆腐」) になります。
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "japanize-matplotlib", "-q"], check=True)
    import japanize_matplotlib  # noqa: F401

# 可視化に使う画像のインデックス。毎回同じ画像で比較するため固定します。
VIS_SAMPLE_INDEX = 0


def _cellbox_to_xyxy(box_cell, i: int, j: int, cell_size: float, img_size: float = IMG_SIZE):
    """
    セル基準の box [x_cell, y_cell, w_norm, h_norm] を、
    画像全体基準の xyxy 座標 (ピクセル) に変換します。
    encode_target で行った変換の、ちょうど逆向きの計算です。
    """
    x_cell, y_cell, w_norm, h_norm = box_cell
    cx = (j + x_cell) * cell_size
    cy = (i + y_cell) * cell_size
    w = max(w_norm, 0.0) * img_size
    h = max(h_norm, 0.0) * img_size
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _iou_xyxy(box1, box2):
    """2つのxyxy boxのIoUを計算します。YoloV1Loss内の計算と同じ考え方です。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def _prepare_vis_sample(dataset, index: int = VIS_SAMPLE_INDEX):
    """
    可視化用に、1枚の画像を学習時と同じ前処理・符号化にかけて取り出します。
    """
    image_tensor, target_dict = dataset[index]
    resized_image, resized_gt_boxes = resize_image_and_boxes(
        image_tensor, target_dict["boxes"], size=IMG_SIZE
    )
    target = encode_target(resized_gt_boxes, target_dict["labels"], img_size=IMG_SIZE, s=S, c=C)
    return resized_image, resized_gt_boxes, target


def visualize_target_encoding(dataset, index: int = VIS_SAMPLE_INDEX):
    """
    【学習前に実行】① グリッド分割と担当セルの確認。

    encode_target が作った教師データを画像に重ねて表示し、
    「GTの中心が入ったセルが担当になっている」というルールが
    正しくコードに反映されているかを目視で確認します。
    """
    resized_image, resized_gt_boxes, target = _prepare_vis_sample(dataset, index)
    cell_size = IMG_SIZE / S
    image_np = (resized_image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    obj_cells = [(i, j) for i in range(S) for j in range(S) if target[i, j, 4] == 1]

    # ---- テキスト出力 ----
    print("\n---------- ① グリッド分割 (教師データの符号化結果) ----------")
    print(f"画像サイズ: {IMG_SIZE}×{IMG_SIZE} / グリッド: {S}×{S} (1セル = {cell_size:.1f}px)")
    print(f"GT box数: {len(resized_gt_boxes)}  →  担当セル数: {len(obj_cells)}")
    if len(resized_gt_boxes) > len(obj_cells):
        print("  ※ GT数より担当セル数が少ないのは、同じセルに複数の顔の中心が入り、")
        print("     v1の「1セル1物体」制約により一部が除外されたためです。")

    for (i, j) in obj_cells[:5]:   # 先頭5件だけ数値を表示します
        x_cell, y_cell, w_norm, h_norm = target[i, j, 0:4].tolist()
        class_id = int(torch.argmax(target[i, j, 5:]).item())
        print(
            f"  セル(i={i}, j={j}): "
            f"x_cell={x_cell:.3f} y_cell={y_cell:.3f} "
            f"w_norm={w_norm:.3f} h_norm={h_norm:.3f} "
            f"class={class_id}(face)"
        )
    if len(obj_cells) > 5:
        print(f"  ... 他 {len(obj_cells) - 5} セル")

    # ---- 画像出力 ----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_np)

    for k in range(S + 1):
        ax.axhline(k * cell_size, color="white", linewidth=1, alpha=0.6)
        ax.axvline(k * cell_size, color="white", linewidth=1, alpha=0.6)

    for (i, j) in obj_cells:
        ax.add_patch(patches.Rectangle(
            (j * cell_size, i * cell_size), cell_size, cell_size,
            linewidth=0, facecolor="yellow", alpha=0.35,
        ))

    for box in resized_gt_boxes:
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none",
        ))

    ax.set_title(f"① グリッド分割 ({S}×{S})\n黄色 = GTの中心が入る担当セル / 緑 = GT box")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def visualize_prediction_detail(model, dataset, epoch: int, index: int = VIS_SAMPLE_INDEX):
    """
    【各エポック終了時に実行】② bbox推定 と ③ クラス予測 の確認。

    担当セルについて、B個の予測boxそれぞれのIoUを計算し、
    「どちらが責任boxとして選ばれたか」を、損失関数と同じロジックで再現して表示します。
    あわせて confidence とクラス確率の数値も出力します。
    """
    model.eval()
    resized_image, resized_gt_boxes, target = _prepare_vis_sample(dataset, index)
    cell_size = IMG_SIZE / S

    pred = model(resized_image.unsqueeze(0).to(DEVICE))[0].cpu()   # [S, S, B*5+C]
    image_np = (resized_image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    obj_cells = [(i, j) for i in range(S) for j in range(S) if target[i, j, 4] == 1]

    print(f"\n---------- ②③ 予測の詳細 (epoch {epoch} 終了時) ----------")

    if len(obj_cells) == 0:
        print("  この画像には担当セルがありません (顔が検出対象外)。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ============================================================
    # ② bbox推定 — 担当セルの責任box判定
    # ============================================================
    ax = axes[0]
    ax.imshow(image_np)

    # 詳細な数値を出す対象は、先頭の担当セル1つに絞ります
    focus_i, focus_j = obj_cells[0]

    for cell_order, (i, j) in enumerate(obj_cells):
        pred_boxes_cell = pred[i, j, : B * 5].reshape(B, 5)
        gt_xyxy = _cellbox_to_xyxy(target[i, j, 0:4].tolist(), i, j, cell_size)

        # ---- 責任box (responsible box) の判定 ----
        # YoloV1Loss と全く同じロジック: GTとのIoUが高い方を責任boxとする
        ious = []
        for b_idx in range(B):
            box_xyxy = _cellbox_to_xyxy(pred_boxes_cell[b_idx, 0:4].tolist(), i, j, cell_size)
            ious.append(_iou_xyxy(box_xyxy, gt_xyxy))
        responsible_idx = int(np.argmax(ious))

        is_focus = (i == focus_i and j == focus_j)

        for b_idx in range(B):
            x1, y1, x2, y2 = _cellbox_to_xyxy(pred_boxes_cell[b_idx, 0:4].tolist(), i, j, cell_size)
            is_responsible = (b_idx == responsible_idx)
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5 if is_responsible else 1.2,
                edgecolor="red" if is_responsible else "orange",
                facecolor="none",
                linestyle="-" if is_responsible else "--",
                alpha=1.0 if is_focus else 0.5,
            ))

        # ---- テキストで数値を出力 (先頭3セルまで) ----
        if cell_order < 3:
            conf_values = pred_boxes_cell[:, 4].tolist()
            class_probs = pred[i, j, B * 5:].tolist()
            print(f"  セル(i={i}, j={j}):")
            for b_idx in range(B):
                mark = "← 責任box" if b_idx == responsible_idx else "   (非責任)"
                print(
                    f"    box{b_idx}: IoU={ious[b_idx]:.3f}  "
                    f"confidence={conf_values[b_idx]:.3f}  {mark}"
                )
            print(f"    クラス確率: {[f'{p:.3f}' for p in class_probs]} (index 0 = face)")
            best_conf = max(conf_values)
            print(f"    → このセルのスコア (confidence×クラス確率) = {best_conf * max(class_probs):.3f}")

    if len(obj_cells) > 3:
        print(f"  ... 他 {len(obj_cells) - 3} セル (画像には全セル分を描画しています)")

    for box in resized_gt_boxes:
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none",
        ))

    ax.set_title(
        f"② bbox推定 (epoch {epoch})\n"
        "実線赤 = 責任box (GTとIoU最大) / 破線橙 = 非責任box / 緑 = GT"
    )
    ax.axis("off")

    # ============================================================
    # ③ クラス予測 — confidence × クラス確率のヒートマップ
    # ============================================================
    ax = axes[1]
    ax.imshow(image_np)

    pred_boxes_all = pred[..., : B * 5].reshape(S, S, B, 5)
    pred_class_all = pred[..., B * 5:]

    score_map = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            class_prob = pred_class_all[i, j].max().item()
            best_conf = pred_boxes_all[i, j, :, 4].max().item()
            score_map[i, j] = best_conf * class_prob

    heatmap = ax.imshow(score_map, extent=(0, IMG_SIZE, IMG_SIZE, 0), cmap="jet", alpha=0.45)
    plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label="confidence × クラス確率")

    ax.set_title(f"③ クラス予測 (epoch {epoch})\n色 = 各セルの「顔らしさ」スコア")
    ax.axis("off")

    print(f"  スコアマップ全体: 最小={score_map.min():.3f}  最大={score_map.max():.3f}")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    """
    検証データに対する平均損失を計算します (勾配計算なし)。
    mAPなどの詳細な評価指標は、この先のステップ (5,6) で別途実装します。
    ここでは「損失が下がっているか」を確認するための簡易チェックです。
    """
    model.eval()
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss, _ = loss_fn(preds, targets)
        running_loss += loss.item()

    return running_loss / len(loader)


# ============================================================
# 5. DataLoaderの作成 (Cell 1 の Dataset を再利用)
# ============================================================
#
# Cell 1 で作った train_dataset / val_easy_dataset を、
# このセルで定義した collate_fn_yolo と組み合わせて使い直します。
#

train_loader_yolo = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_yolo,
)

val_loader_yolo = DataLoader(
    val_easy_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_yolo,
)


# ============================================================
# 6. モデル・損失関数・optimizerの準備
# ============================================================

model = SimpleYoloFace(s=S, b=B, c=C, pretrained=True).to(DEVICE)
loss_fn = YoloV1Loss(s=S, b=B, c=C).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


# ============================================================
# 7. 1バッチだけ動作確認 (本格学習の前に、形が合っているか確認)
# ============================================================
#
# ここで images / targets / preds の形状が想定通りか、
# エラーなく損失が計算できるかを確認してから、下の学習ループに進んでください。

print("\n========== 1バッチ動作確認 ==========")
sample_images, sample_targets = next(iter(train_loader_yolo))
print(f"images shape  : {sample_images.shape}")   # 期待値: [BATCH_SIZE, 3, 448, 448]
print(f"targets shape : {sample_targets.shape}")  # 期待値: [BATCH_SIZE, 7, 7, 6]  (B*5+C=6ではなくtargetは5+C=6)

sample_images = sample_images.to(DEVICE)
sample_targets = sample_targets.to(DEVICE)

model.eval()
with torch.no_grad():
    sample_preds = model(sample_images)
print(f"preds shape   : {sample_preds.shape}")    # 期待値: [BATCH_SIZE, 7, 7, 11]  (B*5+C = 2*5+1 = 11)

sample_loss, sample_logs = loss_fn(sample_preds, sample_targets)
print(f"sample loss   : {sample_logs}")


# ============================================================
# 8. 学習ループ本体
# ============================================================
#
# 動作確認 (手順7) でエラーが出ないことを確認してから、以下を実行してください。
#
# Notebook のセルとして実行される前提のため、Pythonスクリプト向けの
# `if __name__ == "__main__":` は使わず、RUN_TRAINING の値で
# 実行するかどうかを切り替えています。
#
# 補足 (Kaggle Notebookでの使い方):
#   このセル自体が長いため、いったん RUN_TRAINING = False で最後まで実行して
#   モデル定義やエラーの有無を確認したあと、
#   このセルをコピーした新しいセルを作って RUN_TRAINING = True に変更し、
#   学習だけを独立して実行する (途中でエポック数を変えて再実行しやすくする)
#   という使い方もできます。

if RUN_TRAINING:
    # ---- 学習開始前: 教師データの符号化結果を確認 ----
    # ここで「GTの中心が入ったセルが担当になっているか」を目視確認してから
    # 学習に進むことで、符号化のバグに気づかないまま長時間学習してしまう
    # 事態を防げます。
    print("\n========== 学習前の確認: 教師データの符号化 ==========")
    visualize_target_encoding(val_easy_dataset, index=VIS_SAMPLE_INDEX)

    print("\n========== 学習開始 ==========")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader_yolo, optimizer, loss_fn, DEVICE)
        val_loss = evaluate_loss(model, val_loader_yolo, loss_fn, DEVICE)

        print(f"  epoch {epoch} 終了: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ---- エポック終了ごと: 予測の中身を確認 ----
        # 毎回同じ画像 (VIS_SAMPLE_INDEX) を使うため、エポックを追うごとに
        # 責任boxの位置やIoU、confidenceがどう変化していくかを比較できます。
        visualize_prediction_detail(model, val_easy_dataset, epoch=epoch, index=VIS_SAMPLE_INDEX)

        # ---- チェックポイント保存 ----
        # 途中で学習が中断しても再開できるよう、エポックごとに重みを保存します。
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, f"/kaggle/working/yolo_face_epoch{epoch}.pt")

    print("\n========== 学習完了 ==========")
else:
    print("\nRUN_TRAINING = False のため、学習ループはスキップされました。")
    print("動作確認だけを行いたい場合はこのままで問題ありません。")
    print("本格的に学習する場合は RUN_TRAINING = True にして再実行してください。")