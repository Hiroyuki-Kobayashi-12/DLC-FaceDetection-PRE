# ============================================================
# Cell 4: Save Model
# ============================================================

MODEL_PATH = "face_detector_fasterrcnn.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("saved:", MODEL_PATH)