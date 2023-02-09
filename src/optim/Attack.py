import torch
import torch.nn as nn
import torch.optim as optim

upper_limit = 1
lower_limit = 0


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def fgsm(model, inputs,epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(inputs, requires_grad=True)
    
    scores = torch.sigmoid(model(inputs+delta)).squeeze()

    scores.backward()

    
    return epsilon * delta.grad.detach().sign()




def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, epsilon=8/255, alpha=2/255, attack_iters=10, restarts=1, norm="l_inf",c=None,objective=None,R=None):
    max_loss = torch.zeros(X.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)

        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):

            index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss =  getScore(model,X,delta,c,objective,R)
            loss.backward()
            
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        
        all_loss = getScore(model,X,delta,c,objective,R)
        
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta.detach()


def getScore(net,inputs,c=None,objective=None,R=None):
    outputs = net(inputs)
    dist = torch.sum((outputs - c) ** 2, dim=1)
    if objective == 'soft-boundary':
        scores = dist - R ** 2
    else:
        scores = dist
    return scores
