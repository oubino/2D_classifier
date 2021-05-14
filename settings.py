# settings

import torch
from torch.utils.tensorboard import SummaryWriter
import os

def init():
    global root, save_path, save_path_root
    global batch_size, batch_size_test
    global device
    global epochs
    global use_amp
    global classes
    global run_folder
    
    
    root = r'C:\Users\olive\OneDrive\Documents\MPhys\Heart'
    
    batch_size = 100
    batch_size_test = 10
    
    save_path_root = r'C:\Users\olive\OneDrive\Documents\MPhys\Heart\Results'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    classes = ('Normal','Afib')
    
    epochs = 20
    
    use_amp = True#True
    
    run_folder = input ( "run folder (NOTE IT WILL OVERWRITE TENSORBOARD SO MAKE SURE): ")
    print('\n')
    print('run folder')
    print(run_folder)
    input("Press Enter to continue...")
    save_path = os.path.join(save_path_root, run_folder) 
    try:  
        os.mkdir(save_path)  
    except OSError as error:  
            print(error) 
    
def init_tensorboard(fold):
    global writer
    
    # create tensorboard writer
    tensor_folder = os.path.join(save_path_root, 'tensorboard')
    tensorboard_loc = os.path.join(tensor_folder, '%s_fold_%s' % (run_folder,fold))
    writer = SummaryWriter(tensorboard_loc) 

    
