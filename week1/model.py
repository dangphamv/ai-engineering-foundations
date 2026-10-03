from huggingface_hub import PyTorchModelHubMixin
from torch import nn


# The mixin records __init__ kwargs into config.json and adds
# save_pretrained()/from_pretrained(), the same layout real Hub models use.
class MoonsMLP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_size: int = 32, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 2
        for _ in range(num_layers):
            layers += [nn.Linear(in_features, hidden_size), nn.ReLU()]
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
