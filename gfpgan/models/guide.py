import pathlib
import os
from typing import Dict, Tuple
import torch
from torch import nn
from torchvision import transforms


class Guide(nn.Module):
    MODEL_BASE_PATH = pathlib.Path(__file__).parent.resolve()

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def __init__(self, model_path: str, use_logits=False, size: Tuple[int, int]=(224, 224), print_model: bool=False, preprocess=None):
        super(Guide, self).__init__()
        self.preprocess = preprocess
        self.use_logits = use_logits
        self.resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.model: nn.Module = torch.load(os.path.join(self.MODEL_BASE_PATH, model_path))
        if print_model: print(f"############\n{model_path}\n############\n", self.model)
        if use_logits:
            self.activation = {}
            # Get the logits layer from the model
            out = list(self.model.modules())[-2]
            # Add a hook to capture intermediate output
            out.register_forward_hook(self.get_activation('out'))

    def forward(self, x):
        with torch.no_grad():
            if self.preprocess is not None: x = self.preprocess(x)
            x = self.resize(x)
            x = self.model(x)
            if self.use_logits:
                # Return the result from logits so that we can do other operations
                # down the line like crossentropy loss more easily
                return self.activation["out"]
            return x
