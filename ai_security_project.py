import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# ---------------------------------
# DEVICE
# ---------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ---------------------------------
# DATASET
# ---------------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=1000,
    shuffle=False
)

# ---------------------------------
# MODEL
# ---------------------------------

class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        # 28x28 -> 26x26 -> 24x24
        self.fc1 = nn.Linear(32 * 24 * 24, 128)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

model = SimpleCNN().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss()

# ---------------------------------
# TRAIN MODEL
# ---------------------------------

def train(model, loader, epochs=2):

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# ---------------------------------
# EVALUATE MODEL
# ---------------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    print(f"Accuracy: {accuracy*100:.2f}%")

    return accuracy

# ---------------------------------
# FGSM ATTACK
# ---------------------------------

def fgsm_attack(image, epsilon, gradient):

    perturbation = epsilon * gradient.sign()

    adversarial_image = image + perturbation

    adversarial_image = torch.clamp(
        adversarial_image,
        0,
        1
    )

    return adversarial_image

def evaluate_fgsm(model, loader, epsilon=0.25):

    model.eval()

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        images.requires_grad = True

        outputs = model(images)

        loss = criterion(outputs, labels)

        model.zero_grad()

        loss.backward()

        gradient = images.grad.data

        adversarial_images = fgsm_attack(
            images,
            epsilon,
            gradient
        )

        adversarial_outputs = model(adversarial_images)

        predictions = adversarial_outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    print(f"FGSM Accuracy: {accuracy*100:.2f}%")

    return accuracy

# ---------------------------------
# DEFENSE
# ---------------------------------

def adversarial_train(
    model,
    loader,
    epochs=2,
    epsilon=0.25
):

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            images.requires_grad = True

            outputs = model(images)

            loss = criterion(outputs, labels)

            model.zero_grad()

            loss.backward()

            gradient = images.grad.data

            adversarial_images = fgsm_attack(
                images,
                epsilon,
                gradient
            )

            optimizer.zero_grad()

            defended_outputs = model(
                adversarial_images.detach()
            )

            defended_loss = criterion(
                defended_outputs,
                labels
            )

            defended_loss.backward()

            optimizer.step()

            total_loss += defended_loss.item()

        print(f"Defense Epoch {epoch+1} Loss: {total_loss:.4f}")

# ---------------------------------
# MAIN PROGRAM
# ---------------------------------

print("\nTraining baseline model...\n")

train(model, train_loader)

print("\nEvaluating baseline model...\n")

baseline_accuracy = evaluate(
    model,
    test_loader
)

print("\nRunning FGSM attack...\n")

attacked_accuracy = evaluate_fgsm(
    model,
    test_loader
)

print("\nRunning adversarial defense...\n")

adversarial_train(
    model,
    train_loader
)

print("\nEvaluating defended model...\n")

defended_accuracy = evaluate_fgsm(
    model,
    test_loader
)

# ---------------------------------
# FINAL RESULTS
# ---------------------------------

print("\nFINAL RESULTS")
print("-------------------------")

print(
    f"Baseline Accuracy: {baseline_accuracy*100:.2f}%"
)

print(
    f"Under Attack: {attacked_accuracy*100:.2f}%"
)

print(
    f"After Defense: {defended_accuracy*100:.2f}%"
)

# ---------------------------------
# GRAPH
# ---------------------------------

labels = [
    "Baseline",
    "Attack",
    "Defense"
]

accuracies = [
    baseline_accuracy * 100,
    attacked_accuracy * 100,
    defended_accuracy * 100
]

plt.bar(labels, accuracies)

plt.ylabel("Accuracy (%)")

plt.title("AI Security Project Results")
plt.savefig("results/accuracy_graph.png")
plt.show()