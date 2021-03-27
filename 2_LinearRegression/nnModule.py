import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

### 단순 선형 회귀 ###


# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1)

print(list(model.parameters()))

optimizer = optim.SGD(list(model.parameters()), lr = 0.01)

nb_epochs = 3000

for epoch in range(nb_epochs + 1):

    prediction = model(x_train)

    cost = F.mse_loss(prediction,y_train)


    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
      # print(list(model.parameters()))


print(list(model.parameters())[0], "//", list(model.parameters())[1])
print(model(torch.FloatTensor([4.])))


### 다중 선형 회귀 ###

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3,1)

optimizer = optim.SGD(list(model.parameters()), lr = 1e-5)

nb_epochs = 3000

for epoch in range(nb_epochs+1):
    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

x1 = float(input('x1 : '))
x2 = float(input('x2 : '))
x3 = float(input('x3 : '))


new_var = torch.FloatTensor([[x1,x2,x3]])
print(model(new_var))