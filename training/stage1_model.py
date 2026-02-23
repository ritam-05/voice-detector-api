import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Stage1Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        h = self.encoder(x).last_hidden_state
        h = h.mean(dim=1)
        return self.classifier(h).squeeze(1)
