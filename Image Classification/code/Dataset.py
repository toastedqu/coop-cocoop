import torch
import numpy as np 

from config import get_cfg_defaults
cfg = get_cfg_defaults()

def subset_train(image_embeddings,labels,nshot,index,n_classes,per_class=10):
    np.random.seed(1)
    # randomly choose nshot data points from rest of the data
    index_new = np.array([np.random.choice(list(set(j)-set(index)),nshot,replace=False) for j in [np.arange(i,i+per_class) for i in range(0,labels.shape[0],int(labels.shape[0]/n_classes))]]).flatten()
    '''Train set'''
    train_label = labels[index_new]
    train_img_emb = image_embeddings[index_new]
    print("Train set: ",train_img_emb.shape, train_label.shape)

    # randomize train data
    index_new = torch.randperm(train_img_emb.size(0))
    train_img_emb = train_img_emb[index_new]
    train_label = train_label[index_new]
    return [train_img_emb, train_label]


def subset_data(image_embeddings,labels,nshot = 8,test_size = 2,n_classes = len(cfg.DATASET.CLASSNAMES), per_class = 10 ):
  '''
  Function to subset nshots from Flower Dataset
  '''
  # Create Test set
  # randomly choose test_size data points from data
  np.random.seed(1)
  index = np.array([np.random.choice(j,test_size,replace=False) for j in [np.arange(i,i+per_class) for i in range(0,labels.shape[0],int(labels.shape[0]/n_classes))]]).flatten()
  '''Test set'''
  test_label = labels[index]
  test_img_emb = image_embeddings[index]
  print("Test set: ",test_img_emb.shape, test_label.shape)

  if type(nshot) != list:
    train_img_emb, train_label = subset_train(image_embeddings,labels,nshot,index,n_classes,per_class,)
    return (train_img_emb, train_label, test_img_emb,test_label)
  
  else:
    train = {}
    for n in nshot:
      train[n] = subset_train(image_embeddings,labels,n,index,n_classes,per_class)
    return (train,test_img_emb,test_label)
  