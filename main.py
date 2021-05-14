# main_2
import settings
settings.init()

from data_loading import data_loader
from model_ops import model as m
from model_ops import eval
from model_ops import train as t
from model_ops import save_mod as save
from model_ops import initialise_model as init_mod

from sklearn.model_selection import KFold
from torchsummary import summary

kfold = KFold(n_splits = 5, shuffle = True)
    
list_splits = list(kfold.split(data_loader.dataset))

print(list_splits)

init_fold = 0

# model + summary
model_2d_summary = m.classifier()
data_input_shape = (2,128,128)
model_2d_summary.to(settings.device)
summary(model_2d_summary, input_size=(2, 128,128), batch_size = settings.batch_size)

# print fold ids
for fold, (train_ids, test_ids) in enumerate(list_splits[int(init_fold):], int(init_fold)):
    print('------')
    print('fold')
    print(fold)
    print('------')
    print('train ids/test ids')
    print(train_ids, test_ids)
     
# initialise and 
for fold, (train_ids, test_ids) in enumerate(list_splits[int(init_fold):], int(init_fold)):
    # i.e. if init fold is 1 then skips first fold when initialising
    # different dataloader for each fold
    settings.init_tensorboard(fold)
    print('fold')
    print(fold)
    model, optimizer, scheduler, scaler = init_mod.init()
    data_loader.init(fold, train_ids, test_ids)     
    print('Training model + saving model')
    print('--------------')
    model, optimizer, scheduler, scaler, best_loss = t.train_mod(model, optimizer, scheduler, scaler, fold)
    print('Saving model + best loss')
    save.save_model(model, optimizer, scaler, best_loss, settings.save_path, fold)
    print('Evaluating model')
    eval.evaluate(model, settings.classes, fold)
    print('------------')
    exit()
    
"""
# print images
afib = 0
normal = 0
for i, batch in enumerate(data_loader.dataloaders['test']):
    imgs = batch['image']
    labels = batch['label']
    batch_size = imgs.size(0)
    #fig, ax = plt.subplots(figsize=(8, 8))
    #ax.imshow(imgs[b,0])#, vmin=imgs[b,0].min(), vmax=imgs[b,0].max())
    #plt.show()
    for b in range(batch_size):
        if labels[b] == 0:
            normal += 1
        elif labels[b] == 1:
            afib += 1
    print(afib, normal)
print(afib)
print(normal)
"""

