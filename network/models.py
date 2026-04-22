import torch
import torch.nn as nn
import torchvision

class TransferModel(nn.Module):
    def __init__(self, num_out_classes=2, dropout=0.5):
        super(TransferModel, self).__init__()

        # Load pretrained ResNet18
        self.model = torchvision.models.resnet18(pretrained=True)

        # Replace final layer
        num_ftrs = self.model.fc.in_features

        if dropout:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_out_classes)
            )
        else:
            self.model.fc = nn.Linear(num_ftrs, num_out_classes)

    def forward(self, x):
        return self.model(x)


def model_selection(modelname='resnet18', num_out_classes=2, dropout=0.5):
    if modelname == 'resnet18':
        return TransferModel(num_out_classes=num_out_classes, dropout=dropout)
    else:
        raise NotImplementedError("Only resnet18 is supported")