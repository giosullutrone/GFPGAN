from typing import Callable, Tuple
import torch
from gfpgan.models.guide import Guide
import pandas as pd
import numpy as np


def to_0_1(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2

def to_0_255(x: torch.Tensor) -> torch.Tensor:
    return to_0_1(x) * 255

def mean_sub_bgr(x: torch.Tensor) -> torch.Tensor:
    means = torch.tensor([131.0912, 103.8827, 91.4953], device=x.device)
    x = x.flip(1)  # Flip the second dimension (channel dimension)
    return x - means.view(1, 3, 1, 1)  # Broadcasting

def mean_sub_rgb(x: torch.Tensor) -> torch.Tensor:
    means = torch.tensor([91.4953, 103.8827, 131.0912], device=x.device)
    return x - means.view(1, 3, 1, 1)  # Broadcasting

def to_vgg(x: torch.Tensor) -> torch.Tensor:
    return mean_sub_rgb(to_0_255(x))

def build_guiding_loss(device, path_labels: str, age_weight: float=0.1, gender_weight: float=1.0, eth_weight: float=1.0, verbose: bool=False) -> Callable:
    """Build the guiding loss [age, ethnicity and gender]"""
    # L1 loss [age] and negative log-likelihood [gender and ethnicity]
    l1 = torch.nn.L1Loss().to(device)
    cross_gender = torch.nn.CrossEntropyLoss().to(device)
    cross_eth = torch.nn.CrossEntropyLoss().to(device)
    # Init models
    guide_age = Guide(model_path="age_torch.pt", size=(224, 224), print_model=verbose, preprocess=to_vgg).to(device)
    guide_gender = Guide(model_path="gender_torch.pt", size=(224, 224), use_logits=True, print_model=verbose, preprocess=to_vgg).to(device)
    guide_eth = Guide(model_path="eth_torch.pt", size=(224, 224), use_logits=True, print_model=verbose, preprocess=lambda x: to_vgg(x) / 255).to(device)

    # Read dataset labels
    labels = pd.read_csv(path_labels)

    def get_age_gender_eth(gt_path: str) -> Tuple[np.ndarray]:
        r = labels[labels["Path"].str.contains(gt_path)]
        return r["Age"].values, r["Gender"].values, r["Ethnicity"].values

    # ----- Define losses -----
    def loss_age(img_rec, gt_value):
        # The age model needs the image in [0, 255] RGB and the img_rec is in [-1, 1] RGB
        img_rec = to_0_255(img_rec)
        rec = guide_age(img_rec)
        return l1(rec, gt_value)

    def loss_gender(img_rec, gt_value):
        # The gender model needs the image in [0, 255] RGB and the img_rec is in [-1, 1] RGB
        img_rec = to_0_255(img_rec)
        rec = guide_gender(img_rec)
        # CrossEntropyLoss needs the gt as index of the class target, also the output of the model is a softmax but we need a logsoftmax
        return cross_gender(rec, gt_value)

    def loss_eth(img_rec, gt_value):
        # The eth model needs the image in [0, 1] RGB and the img_rec is in [-1, 1] RGB
        img_rec = to_0_1(img_rec)
        rec = guide_eth(img_rec)
        # CrossEntropyLoss needs the gt as index of the class target, also the output of the model is a softmax but we need a logsoftmax
        return cross_eth(rec, gt_value)

    # ----- Define guiding loss -----
    def loss_guide(img_rec, gt_path, loss_dict: dict=None):
        age, gender, eth = get_age_gender_eth(gt_path)
        if verbose: print(f"{gt_path}\n Age: {age}, Gender: {gender}, Eth: {eth}")

        l_age = loss_age(img_rec.clone(), torch.tensor(np.expand_dims(age, axis=0)).to(device)) * age_weight
        l_gender = loss_gender(img_rec.clone(), torch.tensor(gender).to(device)) * gender_weight
        l_eth = loss_eth(img_rec.clone(), torch.tensor(eth).to(device)) * eth_weight
        if loss_dict:
            loss_dict["l_age"] = l_age
            loss_dict["l_gender"] = l_gender
            loss_dict["l_eth"] = l_eth
        return l_age + l_gender + l_eth
    return loss_guide