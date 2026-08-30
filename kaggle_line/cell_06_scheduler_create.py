# Cell 6 Scheduler Create
# ============================================================
#
# このセルの役割:
#
#   Configで選択されたLearning Rate Schedulerを作成し、
#   後続の学習セルへ渡します。
#
# Schedulerはモデル固有にせず、
# どのモデルでも使用できる汎用部品として定義します。
#
# 現在実装しているScheduler:
#
#   - None
#   - StepLR
#   - CosineAnnealingLR
#
# Configから受け取る定数:
#
#   SCHEDULER_NAME
#   NUM_EPOCHS
#   STEP_LR_STEP_SIZE
#   STEP_LR_GAMMA
#   COSINE_ANNEALING_MIN_LR
#
# 前のセルから受け取る変数:
#
#   optimizer
#
# 後続セルへ渡す変数:
#
#   scheduler
#   SCHEDULER_METADATA
#
# Schedulerの更新は、各エポックの学習とValidationが
# 完了したあとにMainセルで1回だけ実行します。
#
# このセルで行わないこと:
#
#   Optimizer作成
#   Scheduler更新
#   学習
#   Validation
#   Learning Rateの履歴保存
#
# ============================================================


import torch


# ============================================================
# None
# ============================================================

def create_no_scheduler(optimizer):
    """Learning RateをSchedulerで変更しない設定を返します。"""
    return None


# ============================================================
# StepLR
# ============================================================

def create_step_lr_scheduler(optimizer):
    """一定エポックごとにLearning Rateを減衰させます。"""
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=STEP_LR_STEP_SIZE,
        gamma=STEP_LR_GAMMA,
    )


# ============================================================
# CosineAnnealingLR
# ============================================================

def create_cosine_annealing_scheduler(optimizer):
    """Learning Rateをコサイン曲線に沿って最小値まで減衰させます。"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=NUM_EPOCHS,
        eta_min=COSINE_ANNEALING_MIN_LR,
    )


# ============================================================
# Scheduler Select
# ============================================================

if SCHEDULER_NAME == "None":
    scheduler = create_no_scheduler(
        optimizer=optimizer
    )

if SCHEDULER_NAME == "StepLR":
    scheduler = create_step_lr_scheduler(
        optimizer=optimizer
    )

if SCHEDULER_NAME == "CosineAnnealingLR":
    scheduler = create_cosine_annealing_scheduler(
        optimizer=optimizer
    )


# ============================================================
# Output
# ============================================================

SCHEDULER_METADATA = {
    "scheduler_name": SCHEDULER_NAME,
}
