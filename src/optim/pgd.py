import torch
import torch.nn as nn

from .attack_torch import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        
        self.model=model

    def forward(self, images,labels,c,R,objective):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)




        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            # outputs = self.get_logits(adv_images)
            cost=getScore(self.model,adv_images,c,R,objective)


            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            # adv_images = adv_images.detach() + self.alpha*grad.sign()
            
            adv_images= adv_images+self.alpha*grad.sign() if labels==0 else adv_images-self.alpha*grad.sign()
            
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            
            adv_images= adv_images+delta if labels==0 else adv_images-delta
            
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    
def getScore(net,inputs,c,R,objective):
    outputs = net(inputs)
    dist = torch.sum((outputs - c) ** 2, dim=1)
    if objective == 'soft-boundary':
        scores = dist - R ** 2
    else:
        scores = dist
    return scores
