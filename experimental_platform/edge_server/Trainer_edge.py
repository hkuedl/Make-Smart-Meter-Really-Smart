import torch
import numpy as np
import random
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 5e-4

def setup_seed(seed: int = 1234):
    """set a fix random seed.

    Args:
        seed (int, optional): random seed. Defaults to 9.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class feature_processor(nn.Module):
    def __init__(self):
        super(feature_processor, self).__init__()
      

        # self.fc_relu1 = torch.nn.Sequential(
        #     nn.Linear(input_size, 16),
        #     torch.nn.ReLU())
        self.fc_relu2 = torch.nn.Sequential(
            nn.Linear(16, 64),
            torch.nn.ReLU())
        self.fc_relu3 = torch.nn.Sequential(
            nn.Linear(64, 128),
            torch.nn.ReLU())
        self.fc_relu4 = torch.nn.Sequential(
            nn.Linear(128, 64),
            torch.nn.ReLU())
        self.fc_relu5 = torch.nn.Sequential(
            nn.Linear(64, 16),
            torch.nn.ReLU())
        # self.linear = nn.Linear(16, 4)
        

    def forward(self, x):
        x = self.fc_relu2(x)
        x = self.fc_relu3(x)
        x = self.fc_relu4(x)
        output = self.fc_relu5(x)
        return output
   
class edge_server():

    def __init__(
            self,
            ):

        self.feature_processor = feature_processor().to(device)
        self.optimizer = torch.optim.Adam(self.feature_processor.parameters(), lr=lr)
        
    def forward(self, act):
        self.act = act.clone().detach()
        pred = self.feature_processor(self.act)       
        return pred.detach().cpu().float().numpy()
    
    def trainer(self, grad, act): 
        self.grad = grad.clone().detach()
        self.act = act.clone().detach().requires_grad_(True)       
        server_act = self.feature_processor(self.act)
        self.feature_processor.train()
        self.optimizer.zero_grad()
        server_act.backward(self.grad)
        self.optimizer.step()
        return self.act.grad.detach().cpu().float().numpy()
    
    def set_model(self, model_params):
        self.feature_processor.load_state_dict(model_params)
        
    def get_model(self):
        return self.feature_processor.state_dict()
