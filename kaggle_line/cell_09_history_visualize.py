# Cell 9 History Visualize
# ============================================================
#
# このセルの役割:
#
#   学習履歴を記録し、JSONとグラフへ出力する
#   汎用関数を作成します。
#
# 記録する内容:
#
#   - Train Loss
#   - Validation Loss
#   - Box Loss
#   - Objectness Loss
#   - Classification Loss
#   - Learning Rate
#
# Configから受け取る定数:
#
#   なし
#
# 前のセルから受け取るもの:
#
#   train_one_epoch()の返り値
#   validate_one_epoch()の返り値
#   optimizer
#
# 後続セルへ渡す関数:
#
#   create_history()
#   add_history()
#   save_history()
#   visualize_history()
#
# 保存先はMainセルから引数で受け取ります。
# このセルではファイル出力を実行しません。
#
# ============================================================


from pathlib import Path
import json

import matplotlib.pyplot as plt


# ============================================================
# History Create
# ============================================================

def create_history():
    """学習履歴を保存する空の辞書を作成します。"""
    return {
        "epoch": [],
        "learning_rate": [],
        "train": [],
        "validation": [],
    }


# ============================================================
# History Add
# ============================================================

def add_history(
    history,
    epoch,
    train_result,
    validation_result,
    optimizer,
):
    """1エポック分の結果とLearning Rateを履歴へ追加します。"""
    history["epoch"].append(epoch)
    history["learning_rate"].append(
        optimizer.param_groups[0]["lr"]
    )
    history["train"].append(dict(train_result))
    history["validation"].append(dict(validation_result))


# ============================================================
# History Save
# ============================================================

def save_history(history, save_path):
    """学習履歴をJSON形式で保存します。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# History Visualize
# ============================================================

def visualize_history(
    history,
    save_path,
    show=False,
):
    """LossとLearning Rateの履歴をグラフへ保存します。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history["epoch"]
    train_history = history["train"]
    validation_history = history["validation"]

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 10),
    )

    # Train / Validation Total Loss
    axes[0, 0].plot(
        epochs,
        [result["loss"] for result in train_history],
        label="Train",
    )
    axes[0, 0].plot(
        epochs,
        [result["loss"] for result in validation_history],
        label="Validation",
    )
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Train Loss Components
    for loss_name, label in (
        ("box_loss", "Box"),
        ("objectness_loss", "Objectness"),
        ("classification_loss", "Classification"),
    ):
        axes[0, 1].plot(
            epochs,
            [result[loss_name] for result in train_history],
            label=label,
        )

    axes[0, 1].set_title("Train Loss Components")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Validation Loss Components
    for loss_name, label in (
        ("box_loss", "Box"),
        ("objectness_loss", "Objectness"),
        ("classification_loss", "Classification"),
    ):
        axes[1, 0].plot(
            epochs,
            [result[loss_name] for result in validation_history],
            label=label,
        )

    axes[1, 0].set_title("Validation Loss Components")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Learning Rate
    axes[1, 1].plot(
        epochs,
        history["learning_rate"],
        label="Learning Rate",
    )
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(figure)
