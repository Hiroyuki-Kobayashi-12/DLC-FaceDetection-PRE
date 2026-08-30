# Cell 5 Optimizer Create
# ============================================================
#
# このセルの役割:
#
#   Configで選択されたOptimizerを作成し、
#   後続の学習セルへ渡します。
#
# Optimizerはモデル固有にせず、
# どのモデルでも使用できる汎用部品として定義します。
#
# 現在実装しているOptimizer:
#
#   - AdamW
#   - Adam
#   - SGD
#
# Configから受け取る定数:
#
#   OPTIMIZER_NAME
#   LEARNING_RATE
#   WEIGHT_DECAY
#   SGD_MOMENTUM
#
# 前のセルから受け取る変数:
#
#   model
#
# 後続セルへ渡す変数:
#
#   optimizer
#   OPTIMIZER_METADATA
#
# ============================================================


import torch


# ============================================================
# AdamW
# ============================================================

def create_adamw_optimizer(model):
    """モデルの学習対象パラメータへAdamWを設定します。"""
    return torch.optim.AdamW(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# Adam
# ============================================================

def create_adam_optimizer(model):
    """モデルの学習対象パラメータへAdamを設定します。"""
    return torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# SGD
# ============================================================

def create_sgd_optimizer(model):
    """モデルの学習対象パラメータへMomentum付きSGDを設定します。"""
    return torch.optim.SGD(
        params=model.parameters(),
        lr=LEARNING_RATE,
        momentum=SGD_MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# Optimizer Select
# ============================================================

if OPTIMIZER_NAME == "AdamW":
    optimizer = create_adamw_optimizer(
        model=model
    )

if OPTIMIZER_NAME == "Adam":
    optimizer = create_adam_optimizer(
        model=model
    )

if OPTIMIZER_NAME == "SGD":
    optimizer = create_sgd_optimizer(
        model=model
    )
                    

# ============================================================
# Output
# ============================================================

OPTIMIZER_METADATA = {
    "optimizer_name": OPTIMIZER_NAME,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "sgd_momentum": SGD_MOMENTUM,
}
