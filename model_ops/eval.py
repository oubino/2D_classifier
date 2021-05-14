# evaluate

# load saved model
import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


import settings
from model_ops import model as m
from data_loading import data_loader



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_loader.dataset.__test__()


def evaluate(model, classes, fold):
    
    load_path = os.path.join(settings.save_path, 'model_%s.pt' % fold)
    model.load_state_dict(torch.load(load_path, map_location = torch.device('cpu')))
    
    # initialise prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cuda')
    lbllist=torch.zeros(0,dtype=torch.long, device='cuda')
    
    with torch.no_grad():
      for data in data_loader.dataloaders['test']:
        images, labels = data['image'].to(device), data['label'].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) # predicted is tensor containing indices of predicted classes
        # Append batch prediction results
        predlist=torch.cat([predlist,predicted.view(-1).to(device)])
        lbllist=torch.cat([lbllist,labels.view(-1).to(device)])
        
    
    # Confusion matrix
    predlist = predlist.cpu()
    lbllist = lbllist.cpu()
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    conf_mat = np.array(conf_mat)
    print(conf_mat)
    
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, cmap="YlGn")
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor", size)
    
    # Loop over data dimensions and create text annotations 
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, conf_mat[i, j], ha="center", va="center", color="r")
    
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.show()
    
    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print('Class' + ' | ' + 'Class Accuracy')
    for i in range(len(classes)):
      print(classes[i] + ' | ' + str(class_accuracy[i]))

