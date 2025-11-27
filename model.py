import torch
import torch.nn as nn
import pytorch_lightning as pl

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, Block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7,
                               stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Block, layers[0], 64, 1)
        self.layer2 = self._make_layer(Block, layers[1], 128, 2)
        self.layer3 = self._make_layer(Block, layers[2], 256, 2)
        self.layer4 = self._make_layer(Block, layers[3], 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, Block, num_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

        layers.append(Block(self.in_channels, out_channels,
                            identity_downsample, stride))
        self.in_channels = out_channels * 4

        for _ in range(num_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels=3, num_classes=1):
    return ResNet(Block, [3, 4, 6, 3], img_channels, num_classes)

class LitRN50(pl.LightningModule):
    def __init__(self, num_classes=1, lr=0.0005):
        super().__init__()
        self.model = ResNet50(img_channels=3, num_classes=num_classes)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x).clamp(0, 100)

    def training_step(self, batch, batch_idx):
        images, ages = batch
        pred = self(images)
        ages = ages.float().unsqueeze(1)
        mae = torch.mean(torch.abs(pred - ages))
        loss = self.loss_fn(pred, ages)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_mae", mae, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, ages = batch
        pred = self(images)
        ages = ages.float().unsqueeze(1)
        mae = torch.mean(torch.abs(pred - ages))
        loss = self.loss_fn(pred, ages)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_mae", mae, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, ages = batch
        pred = self(images)
        ages = ages.float().unsqueeze(1)
        mae = torch.mean(torch.abs(pred - ages))
        loss = self.loss_fn(pred, ages)

        self.log("test_loss", loss)
        self.log("test_mae", mae)
        return loss

    def predict_step(self, batch, batch_idx):
        images, ages = batch
        pred = self(images)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
