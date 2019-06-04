import torch
from torch.nn import functional as F
import numpy as np
x = torch.rand(100, 700)
print(x)
y=torch.rand(100, 700)
print(y)

beta=np.linspace(0, 1, num=100)
#beta=torch.as_tensor(beta_np)
def BBFC_loss(X,Y,beta):
    term1=((beta+1)/beta)
    term2=(X*torch.pow(Y,beta))+(1-X)*torch.pow((1-Y),beta)
    term2=torch.prod(term2, dim=1)
    #print(term2.shape)
    term3=torch.pow(Y,(beta+1))+torch.pow(Y,(beta+1))
    term3=torch.prod(term3, dim=1)
    loss1=torch.sum(term1*term2+term3)
    return loss1
loss2 = F.binary_cross_entropy(x, y, reduction='sum')    
#print(beta)
for i in range(beta.shape[0]):
    loss1=BBFC_loss(x,y,beta[i])
    if i==0:
        all_loss1= loss1.numpy()
    
    print(beta[i])
#plt.axhline(y=.4, xmin=0.25, xmax=0.402, linewidth=2, color = 'k')

#plt.plot(train_loss_list, label="train loss")
    #plt.plot(valid_loss_list, label="validation loss")
    #plt.legend()
    #plt.show()