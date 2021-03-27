import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)


plt.plot(x, y, 'g') # g == green
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)


W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1,requires_grad=True)

hypothesis = 1/(1+torch.exp(-1*(x_train.matmul(W) + b)))

### hypothesis = torch.sigmoid(x_train.matmul(W) + b) ###

print(hypothesis)

losses = -((y_train * torch.log(hypothesis)) + (1-y_train)*(torch.log(1-hypothesis)))
print(losses)

cost = losses.mean()
print(cost)

### cost = F.binary_cross_entropy(hypothesis, y_train) ###


######################################### <<< MODEL >>> #############################################

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer = optim.SGD([W,b], lr = 1)

nb_epochs = 10000

for epoch in range(nb_epochs + 1):
    hypothesis = torch.sigmoid(x_train.matmul(W)+b)

    cost = -(y_train*(torch.log(hypothesis)) + (1-y_train)*torch.log(1-hypothesis)).mean()


    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)