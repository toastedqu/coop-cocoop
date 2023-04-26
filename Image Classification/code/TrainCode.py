import torch
import transformers
import torchvision
import numpy as np


def train(model, data, eval_data, criterion, optimizer, scheduler, n_epoch, device = "cuda" if torch.cuda.is_available() else "cpu"):
  model.train()
  print('-'*20,'Train results','-'*20)
  for epoch in range(n_epoch):
      running_loss = 0.0
      running_corr = 0
      ndata = 0 
      
      for batch in data:
          img_embs, lbls = batch[0].to(device), batch[1].to(device)
          outputs = model(img_embs)
          loss = criterion(outputs, lbls)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          preds = torch.argmax(outputs, dim=1)
          running_loss += loss.item()
          running_corr += torch.sum(preds==lbls.data)
          ndata += img_embs.size(0)

      epoch_loss = running_loss / ndata
      epoch_acc = 100*running_corr.double() / ndata
      if epoch==0 or (epoch+1)%10==0 or epoch==(n_epoch-1):
        print(f"\nEpoch {epoch+1}/{n_epoch}:")
        print('-' * 20)
        print('loss: {:.4f}; acc: {:.4f} %'.format(epoch_loss, np.round(epoch_acc.item(),2)))
      
      scheduler.step()
      
  model.eval()
  running_loss = 0.0
  running_corr = 0
  ndata = 0 

  print('\n','-'*20,'Eval results','-'*20)
  for batch in eval_data:
      img_embs, lbls = batch[0].to(device), batch[1].to(device)
      outputs = model(img_embs)
      loss = criterion(outputs, lbls)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      preds = torch.argmax(outputs, dim=1)
      running_loss += loss.item()
      running_corr += torch.sum(preds==lbls.data)
      ndata += img_embs.size(0)

  epoch_loss = running_loss / ndata
  epoch_acc = (100*running_corr.double() / ndata).item()
  print('loss: {:.4f}; acc: {:.4f}'.format(epoch_loss, np.round(epoch_acc,2)))
  scheduler.step()

  return epoch_acc




# linear probe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def logistic(train,y_train,test,y_test):
  # define the multinomial logistic regression model
  model = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=1000)
  model.fit(train, y_train)

  yhat = model.predict(test)
  acc = accuracy_score(y_test,yhat)
  print("Accuracy: ",np.round(acc,2))
  print("F1-score: ",np.round(f1_score(y_test,yhat,average='macro'),2))

  return acc*100


# zeroShot

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm 
def zeroShot(train_dataloader,clip_model,text_emb):
  acc = []
  f1 = []

  for idx, (img_emb,label) in enumerate(tqdm(train_dataloader)):
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale * img_emb @ text_emb.t()
    logits = logits.max(dim=1)

    acc.append(accuracy_score(logits.indices.detach().cpu().numpy(),label.numpy()))
    f1.append(f1_score(logits.indices.detach().cpu().numpy(),label.numpy(),average='macro'))

  print("\n\nAccuracy: ",np.mean(acc),"\nF1-score: ", np.mean(f1))
  return np.mean(acc)
