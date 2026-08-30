# Cell 8 Validate One Epoch
# ============================================================
#
# このセルの役割:
#
#   val_loaderを1周し、1エポックのValidation Lossを計算する
#   汎用関数を作成します。
#
# モデル固有の出力解釈、target変換、Loss計算は、
# Cell 4で選択されたcriterionが担当します。
#
# Cell 3のval_loaderは難易度別に分割せず、
# valid_bbox=Trueの顔をすべて保持しています。
# target["levels"]も維持されますが、
# このセルでは難易度別の処理を行いません。
#
# Configから受け取る定数:
#
#   なし
#
# 前のセルから受け取るもの:
#
#   model
#   val_loader
#   criterion
#   DEVICE
#
# 後続セルへ渡す関数:
#
#   validate_one_epoch()
#
# ============================================================


import torch


# ============================================================
# Validate One Epoch
# ============================================================

def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
):
    """val_loaderを1周し、1エポックの平均Lossを返します。"""
    model.eval()

    loss_totals = {
        "loss": 0.0,
        "box_loss": 0.0,
        "objectness_loss": 0.0,
        "classification_loss": 0.0,
    }

    with torch.no_grad():
        for images, targets in val_loader:
            images = torch.stack(
                images,
                dim=0,
            ).to(
                device,
                non_blocking=True,
            )

            model_output = model(images)

            loss_result = criterion(
                model_output=model_output,
                targets=targets,
                image_height=images.shape[-2],
                image_width=images.shape[-1],
            )

            for loss_name in loss_totals:
                loss_totals[loss_name] += (
                    loss_result[loss_name]
                    .detach()
                    .item()
                )

    num_batches = len(val_loader)

    return {
        loss_name: loss_total / num_batches
        for loss_name, loss_total in loss_totals.items()
    }
