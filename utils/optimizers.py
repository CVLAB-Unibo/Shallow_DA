import torch

def get_optimizer(network, cfg):
    if cfg.optimizer.optimizer_name == "sgd":
        opt = torch.optim.SGD(network.optim_parameters(cfg.lr), lr=cfg.lr, 
                            momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    else:
        opt = torch.optim.AdamW(network.optim_parameters(cfg.lr), lr=cfg.lr, weight_decay=cfg.optimizer.weight_decay)
    return opt