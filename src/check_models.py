import torch

# Load the .pth file
data = torch.load(r"C:\Users\milge\OneDrive\Dokumenter\Johansen\Bachelor Oppgave\saved_models\training_20241108_095403_focalloss_bilinear_124_B64\best_model.pth")

# Check its type
print(type(data))

# Inspect the contents
if isinstance(data, dict):
    for key, value in data.items():
        print(key, type(value))
else:
    print(data)
