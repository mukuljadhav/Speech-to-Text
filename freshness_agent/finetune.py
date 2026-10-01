# finetune.py
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import ViTForImageClassification, ViTImageProcessor

torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# ── CHANGE 1: Load from expanded_model_v2, not the pkl ──
print("Loading expanded_model_v3...")
old_model = ViTForImageClassification.from_pretrained("models/expanded_model_v3")
processor = ViTImageProcessor.from_pretrained("models/expanded_model_v3")
print(f"✅ Loaded. Existing classes: {old_model.config.num_labels}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

train_dataset = datasets.ImageFolder("my_dataset/train", transform=transform)
val_dataset   = datasets.ImageFolder("my_dataset/val",   transform=transform)

folder_classes = train_dataset.classes
print(f"\nFolders found in my_dataset/train: {len(folder_classes)}")

# ── Merge old labels + new labels ──
existing_labels = list(old_model.config.id2label.values())
all_labels = existing_labels.copy()
for label in folder_classes:
    if label not in all_labels:
        all_labels.append(label)

new_classes = [l for l in all_labels if l not in existing_labels]
print(f"Old classes : {len(existing_labels)}")
print(f"New classes : {len(new_classes)} → {new_classes}")
print(f"Total classes: {len(all_labels)}")

label2id = {label: idx for idx, label in enumerate(all_labels)}
id2label  = {idx: label for idx, label in enumerate(all_labels)}

train_dataset.class_to_idx = {cls: label2id[cls] for cls in folder_classes}
val_dataset.class_to_idx   = {cls: label2id[cls] for cls in folder_classes}

train_dataset.samples = [(path, label2id[folder_classes[old_lbl]])
                         for path, old_lbl in train_dataset.samples]
val_dataset.samples   = [(path, label2id[folder_classes[old_lbl]])
                         for path, old_lbl in val_dataset.samples]

train_dataset.targets = [s[1] for s in train_dataset.samples]
val_dataset.targets   = [s[1] for s in val_dataset.samples]

print(f"\nTrain images: {len(train_dataset)} | Val images: {len(val_dataset)}")

# ── Build new model with expanded head ──
config = old_model.config
config.num_labels = len(all_labels)
config.id2label   = id2label
config.label2id   = label2id

new_model = ViTForImageClassification(config)

# Transfer all matching weights from v2
old_state = old_model.state_dict()
new_state = new_model.state_dict()
transferred = 0
for key in new_state:
    if key in old_state and new_state[key].shape == old_state[key].shape:
        new_state[key] = old_state[key]
        transferred += 1
new_model.load_state_dict(new_state)
print(f"✅ Weights transferred: {transferred} layers preserved from expanded_model_v2")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=0)

device = torch.device("cpu")
new_model.to(device)

# ── CHANGE 2: Unfreeze classifier + last 2 transformer blocks ──
for name, param in new_model.named_parameters():
    param.requires_grad = False

for name, param in new_model.named_parameters():
    if any(x in name for x in [
        "classifier",
        "vit.encoder.layer.11",
        "vit.encoder.layer.10",
        "vit.layernorm",
    ]):
        param.requires_grad = True

trainable = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in new_model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

optimizer = AdamW(
    filter(lambda p: p.requires_grad, new_model.parameters()),
    lr=2e-4
)

# ── CHANGE 3: Save to expanded_model_v3 ──
EPOCHS   = 5
SAVE_DIR = "models/expanded_model_v4"
print(f"\n🚀 Starting training → will save best model to '{SAVE_DIR}'\n")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    new_model.train()
    total_loss, correct, total = 0, 0, 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = new_model(pixel_values=images, labels=labels)

        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()

        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (i + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} | Batch {i+1} | Loss: {outputs.loss.item():.3f}")

    new_model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = new_model(pixel_values=images)
            preds = outputs.logits.argmax(dim=-1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc    = val_correct / val_total * 100
    train_acc  = correct / total * 100
    print(f"\n✅ Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {total_loss:.3f} | "
          f"Train: {train_acc:.1f}% | "
          f"Val: {val_acc:.1f}%")

    # ── CHANGE 4: Save only the best epoch ──
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        new_model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)
        print(f"💾 Best model saved → {SAVE_DIR} (val: {val_acc:.1f}%)\n")
    else:
        print(f"   No improvement (best so far: {best_val_acc:.1f}%)\n")

print(f"🎉 Training complete.")
print(f"   Best val accuracy : {best_val_acc:.1f}%")
print(f"   Model saved to    : {SAVE_DIR}")
print(f"   expanded_model_v3 : untouched ✅")