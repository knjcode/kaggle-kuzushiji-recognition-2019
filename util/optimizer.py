#!/usr/bin/env python
# coding: utf-8

import torch.optim as optim
import adabound


def get_optimizer(args, model, metric_fc):
    if metric_fc:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)
        elif args.optimizer == 'nag':
            optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.base_lr, weight_decay=args.wd)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.base_lr, weight_decay=args.wd, amsgrad=True)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)
        elif args.optimizer == 'adabound':
            optimizer = adabound.AdaBound([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.base_lr, weight_decay=args.wd, final_lr=args.final_lr)
        elif args.optimizer == 'amsbound':
            optimizer = adabound.AdaBound([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.base_lr, weight_decay=args.wd, final_lr=args.final_lr, amsbound=True)
        else:
            raise 'unknown optimizer'
    else:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)
        elif args.optimizer == 'nag':
            optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.wd)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.wd, amsgrad=True)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)
        elif args.optimizer == 'adabound':
            optimizer = adabound.AdaBound(model.parameters(), lr=args.base_lr, weight_decay=args.wd, final_lr=args.final_lr)
        elif args.optimizer == 'amsbound':
            optimizer = adabound.AdaBound(model.parameters(), lr=args.base_lr, weight_decay=args.wd, final_lr=args.final_lr, amsbound=True)
        else:
            raise 'unknown optimizer'


    return optimizer
