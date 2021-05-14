# save model
import torch
import os

def save_model(model, optimizer,scaler, best_loss, save_path, fold):
    model_save_path = os.path.join(save_path, 'model_%s.pt' % fold)
    optimizer_save_path = os.path.join(save_path, 'optimizer_%s.pt' % fold)
    scaler_save_path = os.path.join(save_path, 'scaler_%s.pt' % fold)
    best_loss_save_path = os.path.join(save_path, 'best_loss_%s.pt' % fold)
    
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    #torch.save(scaler.state_dict(), scaler_save_path)
    torch.save(best_loss, best_loss_save_path)
