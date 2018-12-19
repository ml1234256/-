import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

######################### 多项式回归 pytorch 练习 ################################################
######################### plot target function ###################################################
# plt.rcParams['figure.figsize'] = (7, 4)
w_target = np.array([1])
b_target = np.array([0.5])

n_xs = 50
x_sample = np.linspace(-4, 4, n_xs)
y_sample = b_target[0] + w_target[0]*x_sample+ np.sin(x_sample)+ np.random.uniform(-0.5, 0.5, n_xs)
plt.scatter(x_sample, y_sample)
plt.show()
# ##################################################################################################
# ################################## built network #################################################
n = 3 # 多项式项数
x_train = np.stack([x_sample ** i for i in range(1,n+1)], axis=1)
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_sample).float().unsqueeze(1)
# 等价于
# x_sample = np.arrange(0, 5, 1)
# x_sample = torch.from_numpy(x_sample)  # torch.size([5])
# x_train = x_sample.unsqueeze(1)   #  在维度去（行）上增加1行  torch.size([5,1])
# x_train = torch.cat([x_train ** i for i in range(1,4)], 1) # 按行拼接 torch.size([5,3])
x_train = Variable(x_train)
y_train = Variable(y_train)

w = Variable(torch.randn(n, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)
def multi_linear(x, w, b):
    return torch.mm(x, w) + b

# torch.mean(a, dim=0, keepdim=True)
# dim:为0则对a的列求mean,为则对a的行求mean
# keepdim:True为开启对维度求mean
def get_loss(y_, y):
    return torch.mean((y_ - y) ** 2 )

####################### Optimizise #######################################################
epoch = 2000  # 更新次数
a = 0.001  # 学习率
for e in range(epoch):
    y_ = multi_linear(x_train, w, b)
    loss = get_loss(y_, y_train)

    loss.backward()

    w.data -= a*w.grad.data
    b.data -= a*b.grad.data

    w.grad.data.zero_()
    b.grad.data.zero_()

    if (e +1)%20 == 0:
        print('epoch {}:loss {:.5f}'.format(e+1, loss))

print(b.data)
print(w.data)

y_ = multi_linear(x_train, w, b)

plt.scatter(x_train.data.numpy()[:, 0], y_sample, label="real curve", color='b')
plt.plot(x_train.data.numpy()[:, 0], y_.data.numpy(), label="fitting carve", color='r')
plt.legend(loc='upper left')
plt.show()

