import torch.optim as optim

def build_optimizer(model, optim_cfg):
    params = filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adam(params, lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
