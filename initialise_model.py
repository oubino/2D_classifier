# initialise model

import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim

import settings

from model_ops import model as m


def init():
    
    model = m.classifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.01) # use adam lr optimiser
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=settings.use_amp)
    model.to(settings.device)
    
    return model, optimizer, scheduler, scaler
    
    