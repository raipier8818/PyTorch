import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

print(x_train)
print(x_train.shape)

print(y_train)
print(y_train.shape)

W = torch.zeros(1,requires_grad= True)
b = torch.zeros(1,requires_grad= True)

print(W)
print(b)

nb_epoch = 10000
for epoch in range(nb_epoch+1):
    hypothesis = x_train * W + b
#    print(hypothesis)


    cost = torch.mean((hypothesis - y_train)**2)
#    print(cost)

    optimizer = optim.SGD([W,b], lr = 0.001)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch {:4d}/{}   W : {:.3f},  b : {:.3f}   Cost : {:.6f}' .format(epoch, nb_epoch, W.item(), b.item(), cost.item()))

