import torch
import numpy as np
from torch.utils.data import Dataset

# create utilis here

def get_ohe(_Y, num_class = 20):

    target_class = np.zeros([_Y.shape[0], num_class])

    for i in range(target_class.shape[0]):

        target_class[i, int(_Y[i])] = 1

    return target_class

def get_trainValData(path, k=0, spike_ready=True):
    num_class = 20
    bio_len = 150
    # read data
    X_train_icub = torch.FloatTensor(np.load(path + 'icub_train_' + str(k) + '.npy'))
    X_val_icub = torch.FloatTensor(np.load(path + 'icub_val_' + str(k) + '.npy'))
    X_train_bio = torch.FloatTensor(np.load(path + 'bio_train_' + str(k) + '.npy'))
    X_val_bio = torch.FloatTensor(np.load(path + 'bio_val_' + str(k) + '.npy'))
    y_train = torch.FloatTensor(np.load(path + 'labels_train_' + str(k) + '.npy'))
    y_val = torch.FloatTensor(np.load(path + 'labels_val_' + str(k) + '.npy'))
    
    if spike_ready == False:
        return X_train_icub, X_val_icub, X_train_bio, X_val_bio, y_train, y_val
        
    target_class_train = torch.FloatTensor(get_ohe(y_train).reshape(-1, num_class, 1, 1, 1))
    target_class_val = torch.FloatTensor(get_ohe(y_val).reshape(-1, num_class, 1, 1, 1))
    
    X_train_icub = X_train_icub.reshape(X_train_icub.shape[0], 60, 1, 1, X_train_icub.shape[-1])
    X_val_icub = X_val_icub.reshape(X_val_icub.shape[0], 60, 1, 1, X_val_icub.shape[-1])
    
    X_train_bio = X_train_bio.reshape(X_train_bio.shape[0], X_train_bio.shape[1], 1, 1, X_train_bio.shape[-1])
    X_val_bio = X_val_bio.reshape(X_val_bio.shape[0], X_val_bio.shape[1], 1, 1, X_val_bio.shape[-1])
        
    return X_train_icub, X_val_icub,  X_train_bio[...,:bio_len], X_val_bio[...,:bio_len], target_class_train, target_class_val, y_train, y_val

def get_testData(path, spike_ready=True):
    num_class = 20
    bio_len = 150
    X_test_icub = torch.FloatTensor(np.load(path + 'icub_test.npy'))
    X_test_bio = torch.FloatTensor(np.load(path + 'bio_test.npy'))
    y_test = torch.FloatTensor(np.load(path + 'labels_test.npy'))
    if spike_ready == False:
        return X_test_icub, X_test_bio, y_test
    
    X_test_icub = X_test_icub.reshape(X_test_icub.shape[0], 60, 1, 1, X_test_icub.shape[-1])
    X_test_bio = X_test_bio.reshape(X_test_bio.shape[0], X_test_bio.shape[1], 1, 1, X_test_bio.shape[-1])
    target_class_test = torch.FloatTensor(get_ohe(y_test).reshape(-1, num_class, 1, 1, 1))
    
    return X_test_icub, X_test_bio[...,:bio_len], target_class_test, y_test

def get_trainValLoader(path, k=0, spike_ready=True, batch_size=8, shuffle=True):
    
    if spike_ready == False:
        X_train_icub, X_val_icub,  X_train_bio, X_val_bio, y_train, y_val = get_trainValData(path, k, spike_ready)
        train_dataset = torch.utils.data.TensorDataset(X_train_icub, X_train_bio, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size)
    
        val_dataset = torch.utils.data.TensorDataset(X_val_icub, X_val_bio, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset,shuffle=shuffle,batch_size=batch_size)
        
        return train_loader, val_loader, train_dataset, val_dataset
    
    X_train_icub, X_val_icub,  X_train_bio, X_val_bio, target_class_train, target_class_val, y_train, y_val = get_trainValData(path, k, spike_ready)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_icub, X_train_bio, target_class_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size)
    
    val_dataset = torch.utils.data.TensorDataset(X_val_icub, X_val_bio, target_class_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset,shuffle=shuffle,batch_size=batch_size)
    
    return train_loader, val_loader, train_dataset, val_dataset

def get_testLoader(path, spike_ready=True, batch_size=8, shuffle=True):
    
    if spike_ready == False:
        X_test_icub, X_test_bio, y_test = get_testData(path, spike_ready)
    
        test_dataset = torch.utils.data.TensorDataset(X_test_icub, X_test_bio, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=shuffle,batch_size=batch_size)
        return test_loader, test_dataset
        

    X_test_icub, X_test_bio, target_class_test, y_test = get_testData(path, spike_ready)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_icub, X_test_bio, target_class_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=shuffle,batch_size=batch_size)
   
    return test_loader, test_dataset
