import torch.autograd as autograd

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data.cuda(), *args, **kwargs)