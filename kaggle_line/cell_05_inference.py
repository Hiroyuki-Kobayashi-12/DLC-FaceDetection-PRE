# ============================================================
# Cell 5: Inference
# ============================================================

model.eval()

image_tensor, target = val_easy_dataset[0]

with torch.no_grad():

    prediction = model(
        [image_tensor.to(device)]
    )[0]

print(prediction.keys())

print("boxes :", prediction["boxes"].shape)
print("scores:", prediction["scores"].shape)