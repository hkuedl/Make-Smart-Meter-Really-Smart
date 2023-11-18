import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class feature_extractor(nn.Module):
    def __init__(self,
                 input_size=40):
        super(feature_extractor, self).__init__()
        self.input_size = input_size
        
        self.fc_relu1 = torch.nn.Sequential(
            nn.Linear(input_size, 16),
            torch.nn.ReLU())

    def forward(self, x):
        output = self.fc_relu1(x)
        return output

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

class regressor(nn.Module):
    def __init__(self):
        super(regressor, self).__init__()
        
        self.linear = nn.Linear(16, 4)
        
    def forward(self, x):
        output = self.linear(x)
        return output
    

def aggregate_params(global_model, params, alpha):
    aggregated_parameters = {}
    for name, para in global_model.named_parameter():
        if name in params:
            para.data.copy_(alpha * para.data + (1-alpha) * params[name])
            aggregated_parameters[name] = para.data
    return aggregated_parameters