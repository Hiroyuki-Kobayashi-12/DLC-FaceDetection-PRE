# Cell 7 Train One Epoch
# ============================================================
#
# このセルの役割:
#
#   train_loaderを1周し、モデルを1エポック学習する
#   汎用関数を作成します。
#
# モデル固有の出力解釈、target変換、Loss計算は、
# Cell 4で選択されたcriterionが担当します。
#
# Configから受け取る定数:
#
#   なし
#
# 前のセルから受け取るもの:
#
#   model
#   train_loader
#   criterion
#   optimizer
#   DEVICE
#
# 後続セルへ渡す関数:
#
#   train_one_epoch()
#
# ============================================================


import torch


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """train_loaderを1周し、1エポックの平均Lossを返します。"""
    model.train()

    loss_totals = {
        "loss": 0.0,
        "box_loss": 0.0,
        "objectness_loss": 0.0,
        "classification_loss": 0.0,
    }

    for images, targets in train_loader:
        images = torch.stack(
            images,
            dim=0,
        ).to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        model_output = model(images)

        loss_result = criterion(
            model_output=model_output,
            targets=targets,
            image_height=images.shape[-2],
            image_width=images.shape[-1],
        )

        loss_result["loss"].backward()
        optimizer.step()

        for loss_name in loss_totals:
            loss_totals[loss_name] += (
                loss_result[loss_name]
                .detach()
                .item()
            )

    num_batches = len(train_loader)

    return {
        loss_name: loss_total / num_batches
        for loss_name, loss_total in loss_totals.items()
    }
