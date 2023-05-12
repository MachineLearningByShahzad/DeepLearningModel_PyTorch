import torch

"""NN with one neurons"""
# Input tensor
tensor = torch.tensor([[3., 4.], [7, 5]])
print(tensor)


print(tensor.requires_grad)
# In order to track the w , b and allowing autograd pkg to back propagate make this true
tensor.requires_grad_()

print(tensor.requires_grad)

# Simple Neuron
out = tensor * tensor

# Gradient for back pass
out.requires_grad

print(out.grad)

print(out.grad_fn)

out = (tensor * tensor).mean()
print(out.grad_fn)

out.backward()

print(tensor.grad)

with torch.no_grad():
    new_tensor = tensor * tensor

    print("New Tensor: ", new_tensor)
    print("Old tensor gradient: ", tensor.requires_grad)
    print("New tensor gradient: ", new_tensor.requires_grad)

"""NN with two neurons"""




