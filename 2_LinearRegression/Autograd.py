import torch

w = torch.tensor(2.0, requires_grad = True)

y = w**2
z = 2*y + 5

z.backward()

print('{}'.format(w.grad))




w = torch.tensor(2.0, requires_grad = True) # 기울기 저장
b = torch.tensor(1.0, requires_grad = True)

z = (b*w)**2 + b**2

z.backward() # z를 미분한 다음 각 값(w = 2.0, b = 1.0)을 대입하여 w.grad와 b.grad에 저장

print('{}, {}'.format(w.grad, b.grad))