import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss2(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        eps = np.finfo(float).eps
        p_t = torch.where(target == 1, x, 1-x)
        fl = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t + eps)
        fl = torch.where(target == 1, fl * self.alpha, fl * (1 - self.alpha))
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x



class FocalLoss(nn.Module):
    """
        This is an implementation of focal-loss.
        see this webpage for more information: https://amaarora.github.io/2020/06/29/FocalLoss.html
    """
    def __init__(self, alpha = 0.7, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, targets):
        # Flatten all data
        output = output.view(-1)
        targets = targets.view(-1)

        # Find distance to gound truth
        pt = np.array(list(map(lambda p,y: p.item() if y==1 else 1-p.item(), output, targets)))
        pt = np.where(pt == 0, 1e-7, pt)

        # make log
        log = -1*np.log(pt)

        # add gamma and alpha
        focal_loss = ((1-pt)**self.gamma) * log

        if self.alpha != None:
            focal_loss = list(map(lambda x, y: x*self.alpha if y==1 else x*(1-self.alpha) , focal_loss, targets))

        # return focal_loss
        return torch.mean(torch.tensor(focal_loss, requires_grad=True))



# x = torch.tensor([0.1, 0.6, 0.9], requires_grad=True)
# y = torch.tensor([0., 1., 1.], requires_grad=False)

# loss = FocalLoss(alpha=3)
# loss_value = loss.forward(x, y)
# test_loss = nn.BCELoss()
# loss_value = test_loss.forward(x, y)
# print(x.grad, y.grad)
# loss_value.backward()
# print(x.grad, y.grad)


# temp = nn.Softmax()
# X_temp = temp(x)
# print(X_temp.requires_grad)
# Y_temp = temp(y)
# print(Y_temp.requires_grad)