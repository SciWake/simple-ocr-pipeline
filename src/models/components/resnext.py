import torch
from torch import nn


class ResNeXtTwoHead(nn.Module):
    def __init__(self, model_path: str, n_classes: int):
        super().__init__()
        self.model = torch.load(model_path, map_location='cpu')
        self.model.fc = nn.Identity()
        self.doc_type_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
        )
        self.sev_docs_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        features = self.model(x)
        doc_type = self.doc_type_head(features)
        sev_docs = self.sev_docs_head(features)
        return doc_type, sev_docs


class ResNeXt(nn.Module):
    def __init__(self, model_path: str, n_classes: int):
        super().__init__()
        self.model = torch.load(model_path, map_location='cpu')
        self.model.fc = nn.Identity()
        self.doc_type_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        features = self.model(x)
        doc_type = self.doc_type_head(features)
        return doc_type
