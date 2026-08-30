# Cell 4 Loss Create
# ============================================================
#
# このセルの役割:
#
#   Configで選択されたモデルに対応するLossを作成します。
#   モデル出力の解釈、target変換、Loss計算は、
#   モデル専用のLossクラス内へ閉じ込めます。
#
# 現在実装しているLoss:
#
#   - YOLOv5Loss
#
# 設計ルール:
#
#   - モデルごとに独立したLossクラスを作る
#   - 共通親クラスと共通Loss計算関数は作らない
#   - Train / Validationへモデル固有処理を漏らさない
#   - __call__()の入力形式と返り値形式だけを統一する
#   - 例外処理と確認用printは置かない
#
# Configから受け取る定数:
#
#   YOLOV5_BOX_LOSS_WEIGHT
#   YOLOV5_OBJECTNESS_LOSS_WEIGHT
#   YOLOV5_CLASSIFICATION_LOSS_WEIGHT
#   YOLOV5_OBJECTNESS_POSITIVE_WEIGHT
#   YOLOV5_CLASSIFICATION_POSITIVE_WEIGHT
#   YOLOV5_ANCHOR_MATCH_THRESHOLD
#   YOLOV5_LABEL_SMOOTHING
#   YOLOV5_FOCAL_GAMMA
#   YOLOV5_FOCAL_ALPHA
#   YOLOV5_OBJECTNESS_IOU_RATIO
#   YOLOV5_OBJECTNESS_BALANCE
#
# 前のセルから受け取る変数:
#
#   model
#   MODEL_METADATA
#
# 後続セルへ渡す変数:
#
#   criterion
#   LOSS_METADATA
#
# criterionの統一入力:
#
#   criterion(
#       model_output=model_output,
#       targets=targets,
#       image_height=image_height,
#       image_width=image_width,
#   )
#
# criterionの統一返り値:
#
#   {
#       "loss": backwardに使用する合計Loss,
#       "box_loss": 記録用Box Loss,
#       "objectness_loss": 記録用Objectness Loss,
#       "classification_loss": 記録用Classification Loss,
#   }
#
# Cell 3のtarget["levels"]は維持しますが、
# このセルでは難易度によるbbox選択を行いません。
#
# ============================================================


import math

import torch
import torch.nn as nn


# ============================================================
# YOLOv5 Loss
# ============================================================

class YOLOv5Loss:
    """アンカーありYOLOv5のLossを計算する専用クラスです。"""

    def __init__(self, model):
        """モデル、Config、Detect Head情報を保持します。"""
        self.model = model
        self.device = next(model.parameters()).device
        self.detect = model.model[-1]

        self.loss_name = "yolov5_custom_loss"
        self.model_family = "yolov5"
        self.loss_components = (
            "loss",
            "box_loss",
            "objectness_loss",
            "classification_loss",
        )

        # Configから取得するYOLOv5専用Loss設定です。
        self.box_weight = YOLOV5_BOX_LOSS_WEIGHT
        self.objectness_weight = YOLOV5_OBJECTNESS_LOSS_WEIGHT
        self.classification_weight = YOLOV5_CLASSIFICATION_LOSS_WEIGHT
        self.objectness_positive_weight = YOLOV5_OBJECTNESS_POSITIVE_WEIGHT
        self.classification_positive_weight = YOLOV5_CLASSIFICATION_POSITIVE_WEIGHT
        self.anchor_match_threshold = YOLOV5_ANCHOR_MATCH_THRESHOLD
        self.label_smoothing = YOLOV5_LABEL_SMOOTHING
        self.focal_gamma = YOLOV5_FOCAL_GAMMA
        self.focal_alpha = YOLOV5_FOCAL_ALPHA
        self.objectness_iou_ratio = YOLOV5_OBJECTNESS_IOU_RATIO
        self.objectness_balance = YOLOV5_OBJECTNESS_BALANCE

        # Detect Headからモデル構造に依存する値を取得します。
        self.num_classes = self.detect.nc
        self.num_anchors = self.detect.na
        self.num_detection_layers = self.detect.nl
        self.anchors = self.detect.anchors

        self.class_positive_target = 1.0 - 0.5 * self.label_smoothing
        self.class_negative_target = 0.5 * self.label_smoothing

        self.objectness_bce = None
        self.classification_bce = None

    def yolov5_bce_losses(self):
        """ObjectnessとClassificationのBCEを作成します。"""
        self.objectness_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [self.objectness_positive_weight],
                device=self.device,
            ),
            reduction="none",
        )

        self.classification_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [self.classification_positive_weight],
                device=self.device,
            ),
            reduction="none",
        )

    def yolov5_focal_loss(self, loss, prediction, target):
        """Configで有効な場合、BCEへFocal Loss係数を適用します。"""
        if self.focal_gamma == 0.0:
            return loss.mean()

        probability = torch.sigmoid(prediction)
        predicted_probability = (
            target * probability
            + (1.0 - target) * (1.0 - probability)
        )

        alpha_factor = (
            target * self.focal_alpha
            + (1.0 - target) * (1.0 - self.focal_alpha)
        )

        modulating_factor = (
            1.0 - predicted_probability
        ).pow(self.focal_gamma)

        return (
            loss
            * alpha_factor
            * modulating_factor
        ).mean()

    def yolov5_objectness_loss(self, prediction, target):
        """Objectness BCEまたはFocal Lossを計算します。"""
        loss = self.objectness_bce(
            prediction,
            target,
        )

        return self.yolov5_focal_loss(
            loss=loss,
            prediction=prediction,
            target=target,
        )

    def yolov5_classification_loss(self, prediction, target):
        """Classification BCEまたはFocal Lossを計算します。"""
        loss = self.classification_bce(
            prediction,
            target,
        )

        return self.yolov5_focal_loss(
            loss=loss,
            prediction=prediction,
            target=target,
        )

    def yolov5_xyxy_to_xywh(
        self,
        boxes,
        image_height,
        image_width,
    ):
        """リサイズ後xyxyを0から1の中心座標xywhへ変換します。"""
        normalized_boxes = torch.zeros_like(boxes)

        normalized_boxes[:, 0] = (
            (boxes[:, 0] + boxes[:, 2])
            / 2.0
            / float(image_width)
        )

        normalized_boxes[:, 1] = (
            (boxes[:, 1] + boxes[:, 3])
            / 2.0
            / float(image_height)
        )

        normalized_boxes[:, 2] = (
            (boxes[:, 2] - boxes[:, 0])
            / float(image_width)
        )

        normalized_boxes[:, 3] = (
            (boxes[:, 3] - boxes[:, 1])
            / float(image_height)
        )

        return normalized_boxes.clamp(0.0, 1.0)

    def yolov5_image_targets(
        self,
        target,
        image_index,
        image_height,
        image_width,
    ):
        """画像1枚分のtargetを[image, class, x, y, w, h]へ変換します。"""
        boxes = target["boxes"].to(self.device)
        labels = target["labels"].to(self.device)

        if boxes.shape[0] == 0:
            return torch.zeros(
                (0, 6),
                dtype=torch.float32,
                device=self.device,
            )

        normalized_boxes = self.yolov5_xyxy_to_xywh(
            boxes=boxes,
            image_height=image_height,
            image_width=image_width,
        )

        image_indices = torch.full(
            (boxes.shape[0], 1),
            float(image_index),
            dtype=torch.float32,
            device=self.device,
        )

        class_ids = labels.float().unsqueeze(1)

        return torch.cat(
            [image_indices, class_ids, normalized_boxes],
            dim=1,
        )

    def yolov5_batch_targets(
        self,
        targets,
        image_height,
        image_width,
    ):
        """バッチ内の全targetを[M, 6]形式へまとめます。"""
        batch_targets = []

        for image_index, target in enumerate(targets):
            image_targets = self.yolov5_image_targets(
                target=target,
                image_index=image_index,
                image_height=image_height,
                image_width=image_width,
            )

            if image_targets.shape[0] > 0:
                batch_targets.append(image_targets)

        if len(batch_targets) == 0:
            return torch.zeros(
                (0, 6),
                dtype=torch.float32,
                device=self.device,
            )

        return torch.cat(batch_targets, dim=0)

    def yolov5_anchor_targets(self, predictions, targets):
        """正解bboxを各検出層のアンカーとグリッドへ割り当てます。"""
        target_classes = []
        target_boxes = []
        target_indices = []
        target_anchors = []

        num_targets = targets.shape[0]

        gain = torch.ones(
            7,
            device=self.device,
        )

        anchor_indices = torch.arange(
            self.num_anchors,
            device=self.device,
        ).float().view(
            self.num_anchors,
            1,
        ).repeat(
            1,
            num_targets,
        )

        expanded_targets = torch.cat(
            [
                targets.repeat(self.num_anchors, 1, 1),
                anchor_indices[..., None],
            ],
            dim=2,
        )

        offset_scale = 0.5

        offset_directions = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            device=self.device,
        ) * offset_scale

        for layer_index, prediction in enumerate(predictions):
            anchors = self.anchors[layer_index]

            gain[2:6] = torch.tensor(
                [
                    prediction.shape[3],
                    prediction.shape[2],
                    prediction.shape[3],
                    prediction.shape[2],
                ],
                device=self.device,
            )

            layer_targets = expanded_targets * gain

            if num_targets > 0:
                width_height_ratio = (
                    layer_targets[..., 4:6]
                    / anchors[:, None]
                )

                anchor_mask = torch.max(
                    width_height_ratio,
                    1.0 / width_height_ratio,
                ).amax(dim=2) < self.anchor_match_threshold

                layer_targets = layer_targets[anchor_mask]

                grid_xy = layer_targets[:, 2:4]
                inverse_grid_xy = gain[[2, 3]] - grid_xy

                near_left, near_top = (
                    (grid_xy % 1.0 < offset_scale)
                    & (grid_xy > 1.0)
                ).T

                near_right, near_bottom = (
                    (inverse_grid_xy % 1.0 < offset_scale)
                    & (inverse_grid_xy > 1.0)
                ).T

                offset_mask = torch.stack(
                    [
                        torch.ones_like(near_left),
                        near_left,
                        near_top,
                        near_right,
                        near_bottom,
                    ]
                )

                layer_targets = layer_targets.repeat(
                    (5, 1, 1)
                )[offset_mask]

                offsets = (
                    torch.zeros_like(grid_xy)[None]
                    + offset_directions[:, None]
                )[offset_mask]
            else:
                layer_targets = expanded_targets[0]
                offsets = 0

            image_indices, class_ids = layer_targets[:, :2].long().T
            grid_xy = layer_targets[:, 2:4]
            grid_wh = layer_targets[:, 4:6]
            anchor_ids = layer_targets[:, 6].long()

            grid_indices = (grid_xy - offsets).long()
            grid_x, grid_y = grid_indices.T

            grid_x = grid_x.clamp(0, prediction.shape[3] - 1)
            grid_y = grid_y.clamp(0, prediction.shape[2] - 1)

            target_indices.append(
                (
                    image_indices,
                    anchor_ids,
                    grid_y,
                    grid_x,
                )
            )

            target_boxes.append(
                torch.cat(
                    [grid_xy - grid_indices, grid_wh],
                    dim=1,
                )
            )

            target_anchors.append(
                anchors[anchor_ids]
            )

            target_classes.append(class_ids)

        return (
            target_classes,
            target_boxes,
            target_indices,
            target_anchors,
        )

    def yolov5_ciou(self, predicted_boxes, target_boxes):
        """中心座標xywh同士のComplete IoUを計算します。"""
        predicted_xy = predicted_boxes[:, :2]
        predicted_wh = predicted_boxes[:, 2:4].clamp(min=1e-7)
        target_xy = target_boxes[:, :2]
        target_wh = target_boxes[:, 2:4].clamp(min=1e-7)

        predicted_half_wh = predicted_wh / 2.0
        target_half_wh = target_wh / 2.0

        predicted_top_left = predicted_xy - predicted_half_wh
        predicted_bottom_right = predicted_xy + predicted_half_wh
        target_top_left = target_xy - target_half_wh
        target_bottom_right = target_xy + target_half_wh

        intersection_top_left = torch.maximum(
            predicted_top_left,
            target_top_left,
        )
        intersection_bottom_right = torch.minimum(
            predicted_bottom_right,
            target_bottom_right,
        )

        intersection_wh = (
            intersection_bottom_right
            - intersection_top_left
        ).clamp(min=0.0)

        intersection_area = (
            intersection_wh[:, 0]
            * intersection_wh[:, 1]
        )

        predicted_area = predicted_wh[:, 0] * predicted_wh[:, 1]
        target_area = target_wh[:, 0] * target_wh[:, 1]
        union_area = predicted_area + target_area - intersection_area
        iou = intersection_area / union_area.clamp(min=1e-7)

        center_distance = (
            (predicted_xy - target_xy).pow(2).sum(dim=1)
        )

        enclosing_top_left = torch.minimum(
            predicted_top_left,
            target_top_left,
        )
        enclosing_bottom_right = torch.maximum(
            predicted_bottom_right,
            target_bottom_right,
        )
        enclosing_diagonal = (
            (enclosing_bottom_right - enclosing_top_left)
            .pow(2)
            .sum(dim=1)
            .clamp(min=1e-7)
        )

        aspect_ratio_difference = (
            4.0
            / math.pi**2
            * (
                torch.atan(target_wh[:, 0] / target_wh[:, 1])
                - torch.atan(predicted_wh[:, 0] / predicted_wh[:, 1])
            ).pow(2)
        )

        with torch.no_grad():
            aspect_ratio_weight = (
                aspect_ratio_difference
                / (1.0 - iou + aspect_ratio_difference).clamp(min=1e-7)
            )

        return (
            iou
            - center_distance / enclosing_diagonal
            - aspect_ratio_weight * aspect_ratio_difference
        )

    def yolov5_loss(self, predictions, targets):
        """Box、Objectness、Classification Lossを計算します。"""
        box_loss = torch.zeros(1, device=self.device)
        objectness_loss = torch.zeros(1, device=self.device)
        classification_loss = torch.zeros(1, device=self.device)

        (
            target_classes,
            target_boxes,
            target_indices,
            target_anchors,
        ) = self.yolov5_anchor_targets(
            predictions=predictions,
            targets=targets,
        )

        for layer_index, prediction in enumerate(predictions):
            image_indices, anchor_ids, grid_y, grid_x = target_indices[
                layer_index
            ]

            objectness_targets = torch.zeros(
                prediction.shape[:4],
                dtype=prediction.dtype,
                device=self.device,
            )

            num_layer_targets = image_indices.shape[0]

            if num_layer_targets > 0:
                matched_predictions = prediction[
                    image_indices,
                    anchor_ids,
                    grid_y,
                    grid_x,
                ]

                predicted_xy = (
                    matched_predictions[:, :2].sigmoid()
                    * 2.0
                    - 0.5
                )

                predicted_wh = (
                    matched_predictions[:, 2:4].sigmoid()
                    * 2.0
                ).pow(2) * target_anchors[layer_index]

                predicted_boxes = torch.cat(
                    [predicted_xy, predicted_wh],
                    dim=1,
                )

                ciou = self.yolov5_ciou(
                    predicted_boxes=predicted_boxes,
                    target_boxes=target_boxes[layer_index],
                )

                box_loss += (1.0 - ciou).mean()

                objectness_iou = ciou.detach().clamp(min=0.0)
                objectness_targets[
                    image_indices,
                    anchor_ids,
                    grid_y,
                    grid_x,
                ] = (
                    (1.0 - self.objectness_iou_ratio)
                    + self.objectness_iou_ratio * objectness_iou
                ).to(objectness_targets.dtype)

                if self.num_classes > 1:
                    class_targets = torch.full_like(
                        matched_predictions[:, 5:],
                        self.class_negative_target,
                    )

                    class_targets[
                        torch.arange(
                            num_layer_targets,
                            device=self.device,
                        ),
                        target_classes[layer_index],
                    ] = self.class_positive_target

                    classification_loss += self.yolov5_classification_loss(
                        prediction=matched_predictions[:, 5:],
                        target=class_targets,
                    )

            layer_objectness_loss = self.yolov5_objectness_loss(
                prediction=prediction[..., 4],
                target=objectness_targets,
            )

            objectness_loss += (
                layer_objectness_loss
                * self.objectness_balance[layer_index]
            )

        box_loss *= self.box_weight
        objectness_loss *= self.objectness_weight
        classification_loss *= self.classification_weight

        total_loss = (
            box_loss
            + objectness_loss
            + classification_loss
        ) * predictions[0].shape[0]

        return {
            "loss": total_loss,
            "box_loss": box_loss.detach(),
            "objectness_loss": objectness_loss.detach(),
            "classification_loss": classification_loss.detach(),
        }

    def yolov5_predictions(self, model_output):
        """学習・評価モードの出力差を吸収し、生の予測を返します。"""
        if self.model.training:
            return model_output

        return model_output[1]

    def __call__(
        self,
        model_output,
        targets,
        image_height,
        image_width,
    ):
        """モデル出力とtargetから1バッチ分のLossを計算します。"""
        predictions = self.yolov5_predictions(
            model_output=model_output
        )

        yolov5_targets = self.yolov5_batch_targets(
            targets=targets,
            image_height=image_height,
            image_width=image_width,
        )

        return self.yolov5_loss(
            predictions=predictions,
            targets=yolov5_targets,
        )

    def yolov5_result(self):
        """YOLOv5 Lossを全Loss共通の返り値形式へまとめます。"""
        return {
            "criterion": self,
            "loss_name": self.loss_name,
            "model_family": self.model_family,
            "loss_components": self.loss_components,
        }

    def create(self):
        """YOLOv5 Lossの部品を作成し、統一形式で返します。"""
        self.yolov5_bce_losses()

        return self.yolov5_result()


# ============================================================
# Loss Select
# ============================================================

if MODEL_METADATA["model_family"] == "yolov5":
    loss_builder = YOLOv5Loss(model=model)


# ============================================================
# Loss Create
# ============================================================

criterion_result = loss_builder.create()


# ============================================================
# Output
# ============================================================

criterion = criterion_result["criterion"]

LOSS_METADATA = {
    "loss_name": criterion_result["loss_name"],
    "model_family": criterion_result["model_family"],
    "loss_components": criterion_result["loss_components"],
}
