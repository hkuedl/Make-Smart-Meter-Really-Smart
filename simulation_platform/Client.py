import torch
from Data import construct_dataset
from Model import feature_extractor, feature_processor, regressor, full_model, full_Trainer, split_Trainer, proposed_Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

#====================================================
#                Client definition
#====================================================
class Client():

    def __init__(
            self,
            data,
            datas:list,
            lr) -> None:

        self.data = data
        self.datas = datas
        self.train_dataloader, self.val_dataloader, self.test_dataloader, self.scaler, self.input_dim, _, __, ___ = construct_dataset(self.data, self.datas)
        self.train_loss_fn = torch.nn.MSELoss(reduction="mean")
        
        #================================================
        #                Build models
        #================================================
        # 
        # Local
        self.local_model =  full_model(input_size=self.input_dim).to(device)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr)
        self.local_trainer = full_Trainer(model=self.local_model, optimizer=self.local_optimizer, train_loss_fn=self.train_loss_fn)
        
        # FedAvg
        self.fed_model = full_model(input_size=self.input_dim).to(device)
        self.fed_optimizer = torch.optim.Adam(self.fed_model.parameters(), lr=lr)
        self.fed_trainer = full_Trainer(model=self.fed_model, optimizer=self.fed_optimizer, train_loss_fn=self.train_loss_fn)
        
        # Split
        self.split_feature_extractor = feature_extractor(input_size=self.input_dim).to(device)
        self.split_feature_processor = feature_processor().to(device)
        self.split_regressor = regressor().to(device)
        self.split_feature_extractor_optimizer = torch.optim.Adam(self.split_feature_extractor.parameters(), lr=lr)
        self.split_feature_processor_optimizer = torch.optim.Adam(self.split_feature_processor.parameters(), lr=lr)
        self.split_regressor_optimizer = torch.optim.Adam(self.split_regressor.parameters(), lr=lr)
        self.split_trainer = split_Trainer(feature_extractor = self.split_feature_extractor, feature_processor = self.split_feature_processor, regressor = self.split_regressor,
                                     regressor_optimizer = self.split_regressor_optimizer, feature_extractor_optimizer = self.split_feature_extractor_optimizer, 
                                     feature_processor_optimizer = self.split_feature_processor_optimizer, train_loss_fn = self.train_loss_fn)
        
        # SFLV1
        self.sflv1_feature_extractor = feature_extractor(input_size=self.input_dim).to(device)
        self.sflv1_feature_processor = feature_processor().to(device)
        self.sflv1_regressor = regressor().to(device)
        self.sflv1_feature_extractor_optimizer = torch.optim.Adam(self.sflv1_feature_extractor.parameters(), lr=lr)
        self.sflv1_feature_processor_optimizer = torch.optim.Adam(self.sflv1_feature_processor.parameters(), lr=lr)
        self.sflv1_regressor_optimizer = torch.optim.Adam(self.sflv1_regressor.parameters(), lr=lr)
        self.sflv1_trainer = split_Trainer(feature_extractor = self.sflv1_feature_extractor, feature_processor = self.sflv1_feature_processor, regressor =self.sflv1_regressor,
                                            feature_extractor_optimizer = self.sflv1_feature_extractor_optimizer, regressor_optimizer=self.sflv1_regressor_optimizer,
                                            feature_processor_optimizer = self.sflv1_feature_processor_optimizer, train_loss_fn = self.train_loss_fn)
        
        # SFLV2
        self.sflv2_feature_extractor = feature_extractor(input_size=self.input_dim).to(device)
        self.sflv2_feature_processor = feature_processor().to(device)
        self.sflv2_regressor = regressor().to(device)
        self.sflv2_feature_extractor_optimizer = torch.optim.Adam(self.sflv2_feature_extractor.parameters(), lr=lr)
        self.sflv2_feature_processor_optimizer = torch.optim.Adam(self.sflv2_feature_processor.parameters(), lr=lr)
        self.sflv2_regressor_optimizer = torch.optim.Adam(self.sflv2_regressor.parameters(), lr=lr)
        self.sflv2_trainer = split_Trainer(feature_extractor = self.sflv2_feature_extractor, feature_processor = self.sflv2_feature_processor, regressor = self.sflv2_regressor, 
                                         feature_extractor_optimizer = self.sflv2_feature_extractor_optimizer, feature_processor_optimizer = self.sflv2_feature_processor_optimizer, regressor_optimizer = self.sflv2_regressor_optimizer,
                                         train_loss_fn = self.train_loss_fn)

        # Proposed
        self.distillation_feature_extractor = feature_extractor(input_size=self.input_dim).to(device)
        self.distillation_feature_processor = feature_processor().to(device)
        self.distillation_regressor = regressor().to(device)
        self.distillation_auxiliary_regressor = regressor().to(device)
        self.distillation_feature_extractor_optimizer = torch.optim.Adam(self.distillation_feature_extractor.parameters(), lr=lr)
        self.distillation_feature_processor_optimizer = torch.optim.Adam(self.distillation_feature_processor.parameters(), lr=lr)
        self.distillation_regressor_optimizer = torch.optim.Adam(self.distillation_regressor.parameters(), lr=lr)
        self.distillation_auxiliary_regressor_optimizer = torch.optim.Adam(self.distillation_auxiliary_regressor.parameters(), lr=lr)
        self.distillation_trainer = proposed_Trainer(feature_extractor = self.distillation_feature_extractor, feature_processor = self.distillation_feature_processor, regressor = self.distillation_regressor, 
                                                 auxiliary_regressor = self.distillation_auxiliary_regressor, auxiliary_regressor_optimizer = self.distillation_auxiliary_regressor_optimizer,
                                                 feature_extractor_optimizer = self.distillation_feature_extractor_optimizer, feature_processor_optimizer = self.distillation_feature_processor_optimizer, regressor_optimizer = self.distillation_regressor_optimizer,
                                                 train_loss_fn = self.train_loss_fn)
    
    
    #================================================
    #           Model training defination
    #================================================
    # Local
    def local_train(self):
        self.local_trainer.trainer(self.train_dataloader)
    
    # FedAvg
    def fed_train(self):
        self.fed_trainer.trainer(self.train_dataloader)
    
    def fed_finetune(self):
        self.fed_trainer.fine_tune(self.train_dataloader)
    
    def get_fed_model(self):
        return self.fed_model
    
    def set_fed_model(self, model_params):
        self.fed_model.load_state_dict(model_params)

    # Split
    def split_train(self):
        self.split_trainer.split_trainer(self.train_dataloader)
    
    # SFLV1 
    def sflv1_train(self):
        self.sflv1_trainer.federated_split_trainer(self.train_dataloader)
    
    def sflv1_finetune(self):
        self.sflv1_trainer.fine_tune(self.train_dataloader)
    
    def get_sflv1_feature_extractor(self):
        return self.sflv1_feature_extractor
    
    def get_sflv1_feature_processor(self):
        return self.sflv1_feature_processor
    
    def get_sflv1_regressor(self):
        return self.sflv1_regressor
    
    def set_sflv1_feature_extractor(self, model_params):
        self.sflv1_feature_extractor.load_state_dict(model_params)

    def set_sflv1_feature_processor(self, model_params):
        self.sflv1_feature_processor.load_state_dict(model_params)

    def set_sflv1_regressor(self, model_params):
        self.sflv1_regressor.load_state_dict(model_params)
    
    # SFLV2
    def sflv2_finetune(self):
        self.sflv2_trainer.fine_tune(self.train_dataloader)

    def sflv2_train(self, X, Y):
        self.sflv2_trainer.sflv2_trainer(X, Y)
    
    def get_sflv2_feature_extractor(self):
        return self.sflv2_feature_extractor
    
    def set_sflv2_feature_extractor(self, model_params):
        self.sflv2_feature_extractor.load_state_dict(model_params)

    def get_sflv2_feature_processor(self):
        return self.sflv2_feature_processor
    
    def set_sflv2_feature_processor(self, model_params):
        self.sflv2_feature_processor.load_state_dict(model_params)

    def get_sflv2_regressor(self):
        return self.sflv2_regressor
    
    def set_sflv2_regressor(self, model_params):
        self.sflv2_regressor.load_state_dict(model_params)
        
    # Proposed
    def distillation_train(self, mu, gamma):
        self.distillation_trainer.distillation_trainer(self.train_dataloader, mu, gamma)

    def distillation_fine_tune(self):
        self.distillation_trainer.fine_tune(self.train_dataloader)

    def get_distillation_feature_extractor(self):
        return self.distillation_feature_extractor
    
    def set_distillation_feature_extractor(self, model_params):
        self.distillation_feature_extractor.load_state_dict(model_params)

    def get_distillation_feature_processor(self):
        return self.distillation_feature_processor
    
    def set_distillation_feature_processor(self, model_params):
        self.distillation_feature_processor.load_state_dict(model_params)

    def get_distillation_regressor(self):
        return self.distillation_regressor
    
    def set_distillation_regressor(self, model_params):
        self.distillation_regressor.load_state_dict(model_params)

    def get_distillation_auxiliary_regressor(self):
        return self.distillation_auxiliary_regressor
    
    def set_distillation_auxiliary_regressor(self, model_params):
        self.distillation_auxiliary_regressor.load_state_dict(model_params)
        
    

    

