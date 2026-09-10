from model import SpatialTemporalXceptionBiLSTM


model = SpatialTemporalXceptionBiLSTM()

print("\n========== CHECK FREEZE ==========")

backbone_trainable = 0
backbone_frozen = 0

for name, param in model.spatial_extractor.named_parameters():

    if param.requires_grad:
        backbone_trainable += param.numel()
    else:
        backbone_frozen += param.numel()


print(
    f"Xception trainable: "
    f"{backbone_trainable:,}"
)

print(
    f"Xception frozen: "
    f"{backbone_frozen:,}"
)


print("\n========== CHECK TRAIN MODE ==========")

model.train()

print(
    "Xception training:",
    model.spatial_extractor.training
)

print(
    "Bi-LSTM training:",
    model.bi_lstm.training
)

print(
    "Classifier training:",
    model.classifier.training
)