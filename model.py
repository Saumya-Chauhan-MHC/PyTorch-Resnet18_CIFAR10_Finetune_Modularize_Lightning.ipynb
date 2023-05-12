import torch
import torch.nn as nn
from torchvision import models
from pytorch_lightning import LightningModule

from torch.nn import Linear, CrossEntropyLoss, functional as F
from torchmetrics.functional import accuracy

class CIFAR10ResNet(LightningModule):
    def __init__(self, num_classes=10, lr=0.01, momentum=0.9, weight_decay=5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # inputs, targets = batch
        # outputs = self(inputs)
        # loss = self.criterion(outputs, targets)

        _, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        # inputs, targets = batch
        # outputs = self(inputs)
        # loss = self.criterion(outputs, targets)

        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

    def test_step(self, batch, batch_idx):
        # inputs, targets = batch
        # outputs = self(inputs)
        # loss = self.criterion(outputs, targets)
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('test_loss', loss)
        self.log('test_accuracy', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                    momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        return optimizer
