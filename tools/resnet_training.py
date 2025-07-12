import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def train_resnet(
    dataset_path,
    num_classes,
    epochs=100,
    batch_size=32,
    save_model_path="resnet_model_test.onnx",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = torchvision.datasets.ImageFolder(
        root=dataset_path, transform=transform
    )
    print(full_dataset.classes)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total if total > 0 else 0
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )
        model.train()

    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224, device=device),
        save_model_path,
        input_names=["input_1"],
        output_names=["predictions/Softmax"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input_1": {0: "batch_size"},
            "predictions/Softmax": {0: "batch_size"},
        },
    )
    print(f"Model saved to {save_model_path}")


if __name__ == "__main__":
    train_resnet(sys.argv[1], 7, 30)
