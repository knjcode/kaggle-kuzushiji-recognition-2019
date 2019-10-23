#!/usr/bin/env python
# coding: utf-8

import torch.optim as optim


def get_cosine_annealing_lr_scheduler(args, optimizer, max_epoch, iteration):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch * iteration, eta_min=args.cosine_annealing_eta_min)


def get_multi_step_lr_scheduler(args, optimizer, lr_step_epochs, lr_factor):
    return optim.lr_scheduler.MultiStepLR(optimizer, lr_step_epochs, lr_factor)


def get_reduce_lr_on_plateau_scheduler(args, optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True,)
