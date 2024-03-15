import torch 
import torch.nn.functional as F


class AUCMLoss_V1(torch.nn.Module):
    """
    AUC-CL
    """
    def __init__(self, margin=None, imratio=None, device=None):
        super(AUCMLoss_V1, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        #self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.a = torch.ones(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.ones(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
     
    def forward(self, y_pred, y_true):
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1) 

        # V5 Latest
        batch_size = (y_true==1).sum().item()
        posvals = y_pred.masked_select(y_true==1)
        negvals = y_pred.masked_select(y_true==0).reshape(batch_size, -1)
        loss = torch.mean((posvals - 10)**2 - 2*self.alpha*posvals) +\
                torch.mean(torch.sum((negvals - self.b)**2 +\
                2*self.alpha*negvals, dim=-1)) - batch_size
 
        return loss
       
# alias
AUCMLoss = AUCMLoss_V1

