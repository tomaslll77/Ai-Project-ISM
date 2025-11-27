# seno modelio infrastruktura, 
#modelio failiukas trainingui, outputina model.pth
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_utk_dataloaders

class AgeCNN(nn.Module):
    def __init__(self):
        super(AgeCNN, self).__init__()
 
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
 
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
 
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

def train_age_model(
        dataset_path="UTKFace/",
        batch_size=32,
        img_size=224,
        epochs=10,
        lr=0.0005,
        num_workers=0
    ):
 
    # Load training + validation data
    print("Loading dataset...")
    train_loader, val_loader = get_utk_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AgeCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    print(f"Training on {device}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
 
        for images, ages in train_loader:
            images = images.to(device)
            ages = ages.to(device).unsqueeze(1)
 
            optimizer.zero_grad()
 
            outputs = model(images)
            loss = criterion(outputs, ages)
 
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
 
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "age_model.pth")
    print("\nModel saved as age_model.pth")
 
    return model

if __name__ == "__main__":
    train_age_model()