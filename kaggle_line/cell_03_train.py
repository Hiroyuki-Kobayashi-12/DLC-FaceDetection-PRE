# ============================================================
# Cell 3: Training
# ============================================================

import torch

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(NUM_EPOCHS):

    model.train()

    epoch_loss = 0.0

    for images, targets in train_loader:

        images = [
            img.to(device)
            for img in images
        ]

        detection_targets = []

        for t in targets:

            labels = torch.ones(
                (len(t["boxes"]),),
                dtype=torch.int64,
                device=device
            )

            detection_targets.append(
                {
                    "boxes": t["boxes"].to(device),
                    "labels": labels,
                }
            )

        loss_dict = model(
            images,
            detection_targets
        )

        loss = sum(loss_dict.values())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss={epoch_loss:.4f}"
    )

print("Training Finished")