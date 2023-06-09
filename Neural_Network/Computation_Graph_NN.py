import torch
from torch.autograd import Variable

x = Variable(torch.randn(1, 10))
h = Variable(torch.randn(1, 20))
W_h = Variable(torch.randn(20, 20))
W_x = Variable(torch.randn(20, 10))
# h.t() is the transpose of the h variable
h_prod = torch.mm(W_h, h.t())
x_prod = torch.mm(W_x, x.t())

next_h = (h_prod + x_prod).tanh()
loss = next_h.sum()
loss.requires_grad = True
loss.backward()





