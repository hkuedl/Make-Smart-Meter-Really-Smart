from torch import nn
import torch
import torch.utils.data as data
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
    
class full_model(nn.Module):
    def __init__(self,
                 input_size=40):
        super(full_model, self).__init__()
        self.input_size = input_size

        self.fc_relu1 = torch.nn.Sequential(
            nn.Linear(input_size, 16),
            torch.nn.ReLU())
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
        self.linear = nn.Linear(16, 4)
        

    def forward(self, x):
        x = self.fc_relu1(x)
        x = self.fc_relu2(x)
        x = self.fc_relu3(x)
        x = self.fc_relu4(x)
        x = self.fc_relu1(x)
        output = self.linear(x)
        
        return output

class full_model_cen(nn.Module):
    def __init__(self, input_size):
        super(full_model_cen, self).__init__()
      

        self.fc_relu1 = torch.nn.Sequential(
            nn.Linear(input_size, 16),
            torch.nn.ReLU())
        self.linear = nn.Linear(16, 4)
        

    def forward(self, x):
        x = self.fc_relu1(x)
        output = self.linear(x)
        return output

# Model training for split learning-based method
class split_Trainer():

    def __init__(self,
                 feature_extractor: nn.Module,
                 feature_processor: nn.Module,
                 regressor: nn.Module,
                 feature_extractor_optimizer: torch.optim.Optimizer,
                 feature_processor_optimizer: torch.optim.Optimizer,
                 regressor_optimizer: torch.optim.Optimizer,
                 train_loss_fn) -> None:

        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.regressor = regressor
        self.feature_extractor_optimizer = feature_extractor_optimizer
        self.feature_processor_optimizer = feature_processor_optimizer
        self.regressor_optimizer = regressor_optimizer
        self.train_loss_fn = train_loss_fn

    # Compute prediction and gradient 
    def loss_back(
            self,
            feature_processor_output,
            label):
        
        self.feature_processor_output = feature_processor_output
        self.label = label
        self.regressor_optimizer.zero_grad()

        # Compute prediction and loss for regressor
        pred = self.regressor(self.feature_processor_output)
        loss = self.train_loss_fn(pred, self.label)

        # Compute gradient for regressor and split layer
        loss.backward(retain_graph=True)
        split_layer = self.feature_processor_output.grad.clone().detach()
        self.regressor_optimizer.step()

        return split_layer 

    # Model training for Split
    def split_trainer(
            self,
            train_dataloader: data.DataLoader):
        
        self.feature_extractor.train()
        self.feature_processor.train()
        self.regressor.train()
        for X, Y in train_dataloader:
            
            # Forward pass
            feature_extractor_act = self.feature_extractor(X)
            feature_extractor_output = feature_extractor_act.clone().detach().requires_grad_(True)
            feature_processor_act = self.feature_processor(feature_extractor_output)
            feature_processor_output = feature_processor_act.clone().detach().requires_grad_(True)
            
            # Backward pass
            feature_processor_grad = self.loss_back(feature_processor_output, Y)
            self.feature_processor_optimizer.zero_grad()
            feature_processor_act.backward(feature_processor_grad)
            feature_extractor_grad = feature_extractor_output.grad.clone().detach()
            self.feature_processor_optimizer.step()
            self.feature_extractor_optimizer.zero_grad()
            feature_extractor_act.backward(feature_extractor_grad)
            self.feature_extractor_optimizer.step()
    
    # Model training for SFLV1 and SFLV2 in federated round
    def federated_split_trainer(
            self,
            train_dataloader: data.DataLoader):
        
        self.feature_extractor.train()
        self.feature_processor.train()
        self.regressor.train()
        for X, Y in train_dataloader:
            
            # Forward pass
            feature_extractor_act = self.feature_extractor(X)
            feature_extractor_output = feature_extractor_act.clone().detach().requires_grad_(True)
            feature_processor_act = self.feature_processor(feature_extractor_output)
            feature_processor_output = feature_processor_act.clone().detach().requires_grad_(True)
            
            # Backward pass
            feature_processor_grad = self.loss_back(feature_processor_output,Y)
            self.feature_processor_optimizer.zero_grad()
            feature_processor_act.backward(feature_processor_grad)
            self.feature_processor_optimizer.step()
            feature_extractor_grad = feature_extractor_output.grad.clone().detach()
            self.feature_extractor_optimizer.zero_grad()
            feature_extractor_act.backward(feature_extractor_grad)
            self.feature_extractor_optimizer.step()
    
    # Model training for SFLV2 in federated round
    def sflv2_trainer(
            self,
            X,
            Y):
        
        self.feature_extractor.train()
        self.feature_processor.train()
        self.regressor.train()
          
        # Forward pass
        feature_extractor_act = self.feature_extractor(X)
        feature_extractor_output = feature_extractor_act.clone().detach().requires_grad_(True)
        feature_processor_act = self.feature_processor(feature_extractor_output)
        feature_processor_output = feature_processor_act.clone().detach().requires_grad_(True)
        
        # Backward pass
        feature_processor_grad = self.loss_back(feature_processor_output,Y)
        self.feature_processor_optimizer.zero_grad()
        feature_processor_act.backward(feature_processor_grad)
        feature_extractor_grad = feature_extractor_output.grad.clone().detach()
        self.feature_processor_optimizer.step()
        self.feature_extractor_optimizer.zero_grad()
        feature_extractor_act.backward(feature_extractor_grad)
        self.feature_extractor_optimizer.step()

    # Model training for SFLV1 and SFLV2 in fine_tuning round
    def fine_tune(
            self,
            train_dataloader: data.DataLoader):
        
        self.feature_extractor.train()
        self.feature_processor.train()
        self.regressor.train()
        for X, Y in train_dataloader:
            
            # Forward pass
            feature_extractor_act = self.feature_extractor(X)
            feature_extractor_output = feature_extractor_act.clone().detach().requires_grad_(True)
            feature_processor_act = self.feature_processor(feature_extractor_output)
            feature_processor_output = feature_processor_act.clone().detach().requires_grad_(True)
            
            # Backward pass
            feature_processor_grad = self.loss_back(feature_processor_output,Y)
            self.feature_processor_optimizer.zero_grad()
            feature_processor_act.backward(feature_processor_grad)
            feature_extractor_grad = feature_extractor_output.grad.clone().detach()
            self.feature_processor_optimizer.step()
            self.feature_extractor_optimizer.zero_grad()
            feature_extractor_act.backward(feature_extractor_grad)
            self.feature_extractor_optimizer.step()

# Model training for non-split learning-based method            
class full_Trainer:

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loss_fn) -> None:

        self.model = model
        self.train_loss_fn = train_loss_fn
        self.optimizer = optimizer
    
    def trainer(
            self,
            train_dataloader: data.DataLoader):
        
        self.model.train()
        for X, Y in train_dataloader:
            
            # Forward pass
            pred = self.model(X)
            loss = self.train_loss_fn(pred, Y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def fine_tune(
            self,
            train_dataloader: data.DataLoader):
        
        self.model.train()
        for X, Y in train_dataloader:
            
            # Forward pass
            pred = self.model(X)
            loss = self.train_loss_fn(pred, Y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Model training for the proposed method 
class proposed_Trainer():

    def __init__(self,
                 feature_extractor: nn.Module,
                 feature_processor: nn.Module,
                 regressor: nn.Module,
                 auxiliary_regressor: nn.Module,
                 feature_extractor_optimizer: torch.optim.Optimizer,
                 feature_processor_optimizer: torch.optim.Optimizer,
                 regressor_optimizer: torch.optim.Optimizer,
                 auxiliary_regressor_optimizer: torch.optim.Optimizer,
                 train_loss_fn) -> None:

        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.regressor = regressor
        self.auxiliary_regressor = auxiliary_regressor
        self.feature_extractor_optimizer = feature_extractor_optimizer
        self.feature_processor_optimizer = feature_processor_optimizer
        self.regressor_optimizer = regressor_optimizer
        self.auxiliary_regressor_optimizer = auxiliary_regressor_optimizer
        self.train_loss_fn = train_loss_fn

    # Compute prediction and gradient for regressor
    def loss_back(
            self,
            feature_processor_output,
            label):
        
        # Compute prediction and loss for regressor
        self.feature_processor_output = feature_processor_output
        self.label = label
        self.regressor_optimizer.zero_grad()
        pred = self.regressor(self.feature_processor_output)
        pred_regressor = pred.clone().detach().requires_grad_(True)
        loss = self.train_loss_fn(pred, self.label)
        
        # Compute gradient for regressor and split layer
        loss.backward(retain_graph=True)
        split_layer_grad = feature_processor_output.grad.clone().detach()
        self.regressor_optimizer.step()

        return split_layer_grad, pred_regressor 
    
    # Model training for the proposed in federated round 
    def distillation_trainer(
            self,
            train_dataloader: data.DataLoader,
            mu,
            gamma):
        
        self.feature_extractor.train()
        self.feature_processor.train()
        self.regressor.train()
        self.auxiliary_regressor.train()
        for X, Y in train_dataloader:
            # forward pass of the feature extractor
            feature_extractor_act = self.feature_extractor(X)
            feature_extractor_output = feature_extractor_act.clone().detach().requires_grad_(True)
            feature_extractor_output_auxiliary = feature_extractor_act.clone().detach().requires_grad_(True)
            
            # forward pass of the feature processor
            feature_processor_act = self.feature_processor(feature_extractor_output)
            feature_processor_output = feature_processor_act.clone().detach().requires_grad_(True)
            
            # forward and backward pass of the regressor
            feature_processor_grad, teacher_regressor = self.loss_back(feature_processor_output,Y)
            
            # backward pass of the feature processor
            self.feature_processor_optimizer.zero_grad()
            feature_processor_act.backward(feature_processor_grad)
            self.feature_processor_optimizer.step()
            
            # forward and backward pass of the auxiliary regressor
            auxiliary_regressor = self.auxiliary_regressor(feature_extractor_output_auxiliary)   
            loss_regress = self.train_loss_fn(auxiliary_regressor, Y)
            loss_distillation = self.train_loss_fn(auxiliary_regressor, teacher_regressor)
            total_loss = mu * loss_regress + gamma * loss_distillation
            self.auxiliary_regressor_optimizer.zero_grad()
            total_loss.backward()
            feature_extractor_auxiliary_grad = feature_extractor_output_auxiliary.grad.clone().detach()
            self.auxiliary_regressor_optimizer.step()
            self.feature_extractor_optimizer.zero_grad()
            
            # backward pass of feature extractor
            feature_extractor_act.backward(feature_extractor_auxiliary_grad)
            self.feature_extractor_optimizer.step()
        
    # Model training for the proposed in fine-tuning round 
    def fine_tune(
            self,
            train_dataloader: data.DataLoader):
        
        self.feature_extractor.train()
        self.feature_processor.train()
        self.regressor.train()
        self.auxiliary_regressor.train()
        for X, Y in train_dataloader:
            
            # forward pass of the feature extractor
            feature_extractor_act = self.feature_extractor(X)
            feature_extractor_output = feature_extractor_act.clone().detach().requires_grad_(True)
            
            # forward pass of the feature processor
            feature_processor_act = self.feature_processor(feature_extractor_output)
            feature_processor_output = feature_processor_act.clone().detach().requires_grad_(True)
            
            # forward and backward pass of the regressor
            feature_processor_grad, _ = self.loss_back(feature_processor_output,Y)
            
            # backward pass of the feature processor
            self.feature_processor_optimizer.zero_grad()
            feature_processor_act.backward(feature_processor_grad)
            self.feature_processor_optimizer.step()
            
            # backward pass of the feature extractor
            feature_extractor_grad = feature_extractor_output.grad.clone().detach()
            self.feature_extractor_optimizer.zero_grad()
            feature_extractor_act.backward(feature_extractor_grad)
            self.feature_extractor_optimizer.step()
