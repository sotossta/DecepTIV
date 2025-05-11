import torch.nn as nn

class Cross_entropy_Loss(nn.Module):
    def __init__(self):
        super(Cross_entropy_Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions, labels):
        return self.loss(predictions, labels)
