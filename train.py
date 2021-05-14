# train

import torch
import torch.nn as nn
import copy
import time
import numpy as np
from sklearn.metrics import confusion_matrix

from data_loading import data_loader
import settings


def train_mod(model, optimizer, scheduler, scaler, fold):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #weight = torch.tensor([0.45, 0.55]).to(device)
    #criterion = nn.CrossEntropyLoss(weight) # added weight
    criterion = nn.CrossEntropyLoss() # added weight

    epoch_range = settings.epochs

    # implement the training loop
    #train_overall_loss = 100 # arbitrary big number
    #val_running_loss = 100 # arbitrary big number
    best_model_wts = copy.deepcopy(model.state_dict()) 
    #best_loss = 100000000
    best_val_loss = 10000000
    
    class_acc = {}
    for i in range(len(settings.classes)):
        class_acc[i] = 0
    
    for epoch in range(epoch_range):
        print('Epoch {}/{}'.format((epoch + 1), epoch_range))
        print('-' * 10)
        print('Best loss', best_val_loss)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                data_loader.dataset.__train__()
                #scheduler.step()
                print('Learning rates (overall and for sigmas)')
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                data_loader.dataset.__test__()
                print('')
                print('Testing on val set')
                print('')
                model.eval()   # Set model to evaluate mode

            #metrics_total = defaultdict(float)
            #metrics_landmarks = defaultdict(float)
            #for i in settings.classes:
            #  metrics_landmarks[c] = defaultdict(float) 
            # i.e. metrics_landmarks[3]['loss'] is loss for landmark denoted by 3

            imgs_in_set = 0 # i.e. counts total number of images in train or val or test set
            epoch_loss = 0
            for i, batch in enumerate(data_loader.dataloaders[phase]):
                    inputs, labels = batch['image'].to(device), batch['label'].to(device)
                    batch_train_loss = 0

                    # forward
                    # track history only if in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        with torch.cuda.amp.autocast(enabled = settings.use_amp):
                            outputs = model((inputs))
                            loss = criterion(outputs, labels)
                            
                            # add to epoch loss
                            epoch_loss += loss
                            
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update() 
                            scheduler.step()
                            optimizer.zero_grad()
                            
                    batch_train_loss += loss
                    
                    print('train loss', batch_train_loss)
                    
                    # test every 10 batches on val
                    if i % 5 == 4 and phase == 'train':
                        settings.writer.add_scalar('train loss', batch_train_loss, epoch + 1)
                        data_loader.dataset.__test__()
                        model.eval()
                        print('testing every 10 batches')
                        val_loss = 0
                        imgs_tot = 0
                        torch.set_grad_enabled(False)
                        for i, batch in enumerate(data_loader.dataloaders['val']):
                            inputs, labels = batch['image'].to(device), batch['label'].to(device)
                            outputs = model((inputs))
                            loss = criterion(outputs, labels)
                            val_loss += loss
                            imgs_tot += inputs.size(0)
                        val_loss /= imgs_tot
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print('  ' + 'best val loss within train loop: {:4f}'.format(best_val_loss))
                            print('\n')
                            best_model_wts = copy.deepcopy(model.state_dict())
                        settings.writer.add_scalar('val loss', val_loss, epoch + 1)
                        
                        print('val loss', val_loss)
                    

                    # statistics
                    imgs_in_set += inputs.size(0)
                    
            print('Images in set')    
            print(imgs_in_set)

            #print('')
            #print('Summary on %s dataset' % phase)
            #print('')
            #functions.print_metrics(metrics_landmarks, imgs_in_set, phase)
            # print metrics divides the values by number of images
            # i.e. when values added to metrics it creates a total sum over all images in the set
            # within print metrics it divides by the total number of images so all values are means
            
            #for l in S.landmarks:
            #    epoch_loss += metrics_landmarks[l]['loss'] # total loss i.e. each batch loss summed
                
                # add loss per landmark to tensorboard
            #if phase == 'train': 
            #S.writer.add_scalar('sigma for landmark %1.0f' % l, sigmas[l][0].item(),epochs_completed + epoch + 1)
            #print('writing to tensorboard')
            
        
            """
            # deep copy the model
            # note validation is done NOT using sliding window
            if phase == 'val' and epoch_loss < best_val_loss:
                print("\n")
                print("------ deep copy best model (end epoch) ------ ")
                #best_loss = epoch_loss
                best_val_loss = best_loss
                print('  ' + 'best val loss: {:4f}'.format(best_val_loss))
                print('\n')
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val':
                _, predicted = torch.max(outputs, 1) # predicted is tensor containing indices of predicted classes
                # Append batch prediction results
                predlist=torch.zeros(0,dtype=torch.long, device='cuda')
                lbllist=torch.zeros(0,dtype=torch.long, device='cuda')
                predlist=torch.cat([predlist,predicted.view(-1).to(device)])
                lbllist=torch.cat([lbllist,labels.view(-1).to(device)])
                conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
                conf_mat = np.array(conf_mat)
                class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
                save = True
                for i in range(len(settings.classes)):
                    if class_accuracy[i] < class_acc[i]:
                        save = False
                if save == True:
                    best_model_wts = copy.deepcopy(model.state_dict())
                """
        time_elapsed = time.time() - since
        finish_time = time.ctime(time_elapsed * (epoch_range - epoch) + time.time())
        print('\n')
        print('Epoch time: ' + '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Estimated finish time (till end of epoch batch): ', finish_time)
        print('\n')
        # save number of epochs completed
        #epochs_completed_total = epochs_completed + epoch + 1
        
    model.load_state_dict(best_model_wts)
    return model, optimizer, scheduler, scaler, best_val_loss
         
            
         
            
