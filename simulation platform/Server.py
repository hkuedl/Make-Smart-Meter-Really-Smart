import torch 
from torch import nn
import numpy as np
from Data import construct_dataset
from Model import feature_processor, feature_extractor, regressor, full_model, full_model_cen, full_Trainer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import math
device = "cuda" if torch.cuda.is_available() else "cpu"

# sMAPE calculation
def symmetric_mean_absolute_percentage_error(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    smape_value = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    return smape_value


#====================================================
#                Server definition
#====================================================
class Server():

    def __init__(self,
                 data,
                 datas,
                 clients,
                 lr,
                 rounds,
                 mu,
                 gamma) -> None:
        
        self.data = data
        self.datas = datas
        self.lr = lr
        self.clients = clients
        _, __, ___, self.scaler, self.input_dim, self.global_train_dataloader, self.global_val_dataloader, self.global_test_dataloader = construct_dataset(self.data, self.datas)
        self.rounds = rounds
        self.mu = mu
        self.gamma = gamma
        self.train_loss_fn = torch.nn.MSELoss(reduction="mean")

    #====================================================
    #                Test model performance
    #====================================================
    # Centralized
    def cen_model_test(self, model):

        self.model = model
        aver_mape = float()
        aver_mse = float()
        aver_mae = float()
        total_mape = float()
        total_mse = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.model.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            pred = self.model(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            y_test = scalery.inverse_transform(y_test)
            mape = mean_absolute_percentage_error(pred, y_test) * 100
            mae = mean_absolute_error(pred, y_test)
            mse = mean_squared_error(pred, y_test)
            total_mape = total_mape + mape
            total_mae = total_mae + mae
            total_mse = total_mse + mse
        aver_mse = total_mse/len(self.clients)
        aver_mape = total_mape/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_mse, aver_mape, aver_mae
    # Local
    def local_model_test(self):
        
        aver_mape = float()
        aver_mse = float()
        aver_mae = float()
        total_mape = float()
        total_mse = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].local_model.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            pred = self.clients[id].local_model(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            y_test = scalery.inverse_transform(y_test)
            
            mape = mean_absolute_percentage_error(pred, y_test) * 100  
            mae = mean_absolute_error(pred, y_test)
            mse = mean_squared_error(pred, y_test)
            total_mape = total_mape + mape
            total_mae = total_mae + mae
            total_mse = total_mse + mse
        aver_mse = total_mse/len(self.clients)
        aver_mape = total_mape/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_mse, aver_mape, aver_mae  

    # FedAvg
    def fed_model_test(self):
        aver_mape = float()
        aver_mse = float()
        aver_mae = float()
        total_mape = float()
        total_mse = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].fed_model.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            pred = self.clients[id].fed_model(x_test).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            y_test = scalery.inverse_transform(y_test)
            
            mape = mean_absolute_percentage_error(pred, y_test) * 100
            mae = mean_absolute_error(pred, y_test)
            mse = mean_squared_error(pred, y_test)
            total_mape = total_mape + mape
            total_mae = total_mae + mae
            total_mse = total_mse + mse
        aver_mse = total_mse/len(self.clients)
        aver_mape = total_mape/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_mse, aver_mape, aver_mae 
    
    # Split
    def split_model_test(self):
        aver_mape = float()
        aver_mse = float()
        aver_mae = float()
        total_mape = float()
        total_mse = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].split_feature_extractor.eval()
            self.clients[id].split_feature_processor.eval()
            self.clients[id].split_regressor.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            pred = self.clients[id].split_feature_extractor(x_test)
            pred = self.clients[id].split_feature_processor(pred)
            pred = self.clients[id].split_regressor(pred).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            y_test = scalery.inverse_transform(y_test)
            
            mape = mean_absolute_percentage_error(pred, y_test) * 100
            mae = mean_absolute_error(pred, y_test)
            mse = mean_squared_error(pred, y_test)
            total_mape = total_mape + mape
            total_mae = total_mae + mae
            total_mse = total_mse + mse
        aver_mse = total_mse/len(self.clients)
        aver_mape = total_mape/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_mse, aver_mape, aver_mae
    
    # SFLV1
    def sflv1_model_test(self):
        aver_mape = float()
        aver_mse = float()
        aver_mae = float()
        total_mape = float()
        total_mse = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].sflv1_feature_extractor.eval()
            self.clients[id].sflv1_feature_processor.eval()
            self.clients[id].sflv1_regressor.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            pred = self.clients[id].sflv1_feature_extractor(x_test)
            pred = self.clients[id].sflv1_feature_processor(pred)
            pred = self.clients[id].sflv1_regressor(pred).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            y_test = scalery.inverse_transform(y_test)
            
            mape = mean_absolute_percentage_error(pred, y_test) * 100
            mae = mean_absolute_error(pred, y_test)
            mse = mean_squared_error(pred, y_test)
            total_mape = total_mape + mape
            total_mae = total_mae + mae
            total_mse = total_mse + mse
        aver_mse = total_mse/len(self.clients)
        aver_mape = total_mape/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_mse, aver_mape, aver_mae
    
    # SFLV2
    def sflv2_model_test(self):
        aver_mape = float()
        aver_mse = float()
        aver_mae = float()
        total_mape = float()
        total_mse = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].sflv2_feature_extractor.eval()
            self.clients[id].sflv2_feature_processor.eval()
            self.clients[id].sflv2_regressor.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            pred = self.clients[id].sflv2_feature_extractor(x_test)
            pred = self.clients[id].sflv2_feature_processor(pred)
            pred = self.clients[id].sflv2_regressor(pred).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            y_test = scalery.inverse_transform(y_test)
            
            mape = mean_absolute_percentage_error(pred, y_test) * 100
            mae = mean_absolute_error(pred, y_test)
            mse = mean_squared_error(pred, y_test)
            total_mape = total_mape + mape
            total_mae = total_mae + mae
            total_mse = total_mse + mse
        aver_mse = total_mse/len(self.clients)
        aver_mape = total_mape/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_mse, aver_mape, aver_mae
    
    # Proposed
    def distillation_model_test(self):
        aver_mape = float()
        aver_mse = float()
        aver_mae = float()
        total_mape = float()
        total_mse = float()
        total_mae = float()
        for id in range(0,len(self.clients)):
            self.clients[id].distillation_feature_extractor.eval()
            self.clients[id].distillation_feature_processor.eval()
            self.clients[id].distillation_regressor.eval()
            client_test_dataloader = self.clients[id].test_dataloader
            x_test, y_test = next(iter(client_test_dataloader))
            pred = self.clients[id].distillation_feature_extractor(x_test)
            pred = self.clients[id].distillation_feature_processor(pred)
            pred = self.clients[id].distillation_regressor(pred).cpu().detach().numpy()
            y_test = y_test.cpu().detach().numpy()
            scalery = self.clients[id].scaler
            pred = scalery.inverse_transform(pred)
            y_test = scalery.inverse_transform(y_test)
            
            mape = mean_absolute_percentage_error(pred, y_test) * 100
            mae = mean_absolute_error(pred, y_test)
            mse = mean_squared_error(pred, y_test)
            total_mape = total_mape + mape
            total_mae = total_mae + mae
            total_mse = total_mse + mse
        aver_mse = total_mse/len(self.clients)
        aver_mape = total_mape/len(self.clients)
        aver_mae = total_mae/len(self.clients)
        return aver_mse, aver_mape, aver_mae

    
    #====================================================
    #                Model aggregation
    #====================================================
    # FedAvg
    def fed_model_average(self):
        models = []

        for client in self.clients:
            model = client.get_fed_model()
            models.append(model)
        
        avg_model_params = models[0].state_dict()

        for param_name in avg_model_params:
            for i in range(1, len(models)):
                avg_model_params[param_name] += models[i].state_dict()[param_name]
            avg_model_params[param_name] /= len(models)
        
        return avg_model_params
    
    # SFLV1
    def sflv1_model_average(self):
        feature_extractors = []
        feature_processors = []
        regressors = []
        
        for client in self.clients:
            feature_extractor = client.get_sflv1_feature_extractor()
            feature_extractors.append(feature_extractor)
            regressor = client.get_sflv1_regressor()
            regressors.append(regressor)
        for server in self.clients:
            feature_processor = server.get_sflv1_feature_processor()
            feature_processors.append(feature_processor)
        
        avg_feature_extractor_params = feature_extractors[0].state_dict()
        avg_feature_processor_params = feature_processors[0].state_dict()
        avg_regressor_params = regressors[0].state_dict()

        for param_name in avg_feature_extractor_params:
            for i in range(1, len(feature_extractors)):
                avg_feature_extractor_params[param_name] += feature_extractors[i].state_dict()[param_name]
            avg_feature_extractor_params[param_name] /= len(feature_extractors)
        for param_name in avg_feature_processor_params:
            for i in range(1, len(feature_processors)):
                avg_feature_processor_params[param_name] += feature_processors[i].state_dict()[param_name]
            avg_feature_processor_params[param_name] /= len(feature_processors)
        for param_name in avg_regressor_params:
            for i in range(1, len(regressors)):
                avg_regressor_params[param_name] += regressors[i].state_dict()[param_name]
            avg_regressor_params[param_name] /= len(regressors)
        
        return avg_feature_extractor_params, avg_feature_processor_params, avg_regressor_params   
    
    # SFLV2_end_side
    def sflv2_extractor_regressor_average(self):
        feature_extractors = []
        regressors = []

        for client in self.clients:
            feature_extractor = client.get_sflv2_feature_extractor()
            feature_extractors.append(feature_extractor)
            regressor = client.get_sflv2_regressor()
            regressors.append(regressor)
        
        avg_feature_extractor_params = feature_extractors[0].state_dict()
        avg_regressor_params = regressors[0].state_dict()

        for param_name in avg_feature_extractor_params:
            for i in range(1, len(feature_extractors)):
                avg_feature_extractor_params[param_name] += feature_extractors[i].state_dict()[param_name]
            avg_feature_extractor_params[param_name] /= len(feature_extractors)

        for param_name in avg_regressor_params:
            for i in range(1, len(regressors)):
                avg_regressor_params[param_name] += regressors[i].state_dict()[param_name]
            avg_regressor_params[param_name] /= len(regressors)
        
        return avg_feature_extractor_params, avg_regressor_params
    
    # SFLV2_edge_side
    def sflv2_processor_average(self):
        feature_processors = []

        for client in self.clients:
            feature_processor = client.get_sflv2_feature_processor()
            feature_processors.append(feature_processor)
        
        avg_feature_processor_params = feature_processors[0].state_dict()

        for param_name in avg_feature_processor_params:
            for i in range(1, len(feature_processors)):
                avg_feature_processor_params[param_name] += feature_processors[i].state_dict()[param_name]
            avg_feature_processor_params[param_name] /= len(feature_processors)
        
        return avg_feature_processor_params
    
    # Proposed_end_edge
    def distillation_extractor_regressor_auxiliary_average(self):
        feature_extractors = []
        regressors = []
        auxiliary_regressors = []

        for client in self.clients:
            feature_extractor = client.get_distillation_feature_extractor()
            feature_extractors.append(feature_extractor)
            regressor = client.get_distillation_regressor()
            regressors.append(regressor)
            auxiliary_regressor= client.get_distillation_auxiliary_regressor()
            auxiliary_regressors.append(auxiliary_regressor)
        
        avg_feature_extractor_params = feature_extractors[0].state_dict()
        avg_regressor_params = regressors[0].state_dict()
        avg_auxiliary_regressor_params = auxiliary_regressors[0].state_dict()

        for param_name in avg_feature_extractor_params:
            for i in range(1, len(feature_extractors)):
                avg_feature_extractor_params[param_name] += feature_extractors[i].state_dict()[param_name]
            avg_feature_extractor_params[param_name] /= len(feature_extractors)

        for param_name in avg_regressor_params:
            for i in range(1, len(regressors)):
                avg_regressor_params[param_name] += regressors[i].state_dict()[param_name]
            avg_regressor_params[param_name] /= len(regressors)
        
        for param_name in avg_auxiliary_regressor_params:
            for i in range(1, len(auxiliary_regressors)):
                avg_auxiliary_regressor_params[param_name] += auxiliary_regressors[i].state_dict()[param_name]
            avg_auxiliary_regressor_params[param_name] /= len(auxiliary_regressors)
        
        return avg_feature_extractor_params, avg_regressor_params, avg_auxiliary_regressor_params
    
    # Proposed_edge_cloud
    def distillation_processor_average(self):
        feature_processors = []

        for client in self.clients:
            feature_processor = client.get_distillation_feature_processor()
            feature_processors.append(feature_processor)
        
        avg_feature_processor_params = feature_processors[0].state_dict()

        for param_name in avg_feature_processor_params:
            for i in range(1, len(feature_processors)):
                avg_feature_processor_params[param_name] += feature_processors[i].state_dict()[param_name]
            avg_feature_processor_params[param_name] /= len(feature_processors)
        
        return avg_feature_processor_params

    #====================================================
    #                Model training
    #====================================================
    # Centralized
    def centralized_train(self):
        print('Centralized Training!')
        self.centralized_model = full_model_cen(input_size=self.input_dim).to(device)
        for e in range(self.rounds):
            self.optimizer = torch.optim.Adam(self.centralized_model.parameters(), lr=self.lr)
            self.trainer = full_Trainer(model=self.centralized_model,
                                optimizer=self.optimizer, train_loss_fn=self.train_loss_fn)
            self.trainer.trainer(self.global_train_dataloader)

        mse, mape, mae = self.cen_model_test(self.centralized_model)
        rmse = math.sqrt(mse)
        print(f"rmse: {rmse}, mape: {mape}, mae: {mae}")
    
    # Local
    def local_train(self):
        print('Local Training!')
        for e in range(self.rounds):
            for client in self.clients:
                client.local_train()

        mse, mape, mae = self.local_model_test()
        rmse = math.sqrt(mse)
        print(f"rmse: {rmse}, mape: {mape}, mae: {mae}")

    # FedAvg
    def fed_train(self):
        print('Federated Training!')
        self.fed_global_model = full_model(input_size=self.input_dim).to(device)

        # local train
        for e in range(self.rounds):
            for client in self.clients:
                client.fed_train()

            # Model average   
            fed_global_model_params = self.fed_model_average()
            self.fed_global_model.load_state_dict(fed_global_model_params)

            # Model distribute
            for client in self.clients:
                client.set_fed_model(fed_global_model_params)
        
        for i in range(30):
            for client in self.clients:
                client.fed_finetune()

        mse, mape, mae = self.fed_model_test()
        rmse = math.sqrt(mse)
        print(f"rmse: {rmse}, mape: {mape}, mae: {mae}")
    
    # Split
    def split_train(self):
        print('Split Training!')

        # train
        for e in range(self.rounds):
            for client in self.clients:
                client.split_train()
            
        mse, mape, mae = self.split_model_test()
        rmse = math.sqrt(mse)
        print(f"rmse: {rmse}, mape: {mape}, mae: {mae}")
    
    # SFLV1
    def sflv1_train(self):
        print('SFLV1 Training!')
        self.global_feature_extractor = feature_extractor(input_size=self.input_dim).to(device)
        self.global_feature_processor = feature_processor().to(device)
        self.global_regressor = regressor().to(device)

        # train
        for e in range(self.rounds):
            for client in self.clients:
                client.sflv1_train()

            # Model average   
            global_feature_extractor_params, global_feature_processor_params, global_regressor_params  = self.sflv1_model_average()
            self.global_feature_extractor.load_state_dict(global_feature_extractor_params)
            self.global_feature_processor.load_state_dict(global_feature_processor_params)
            self.global_regressor.load_state_dict(global_regressor_params)

            # Model distribute
            for client in self.clients:
                client.set_sflv1_feature_extractor(global_feature_extractor_params)
                client.set_sflv1_regressor(global_regressor_params)
                
            for server in self.clients:
                server.set_sflv1_feature_processor(global_feature_processor_params)
        
        # Model Finetune
        for i in range(30):
            for client in self.clients:
                client.sflv1_finetune()

        mse, mape, mae = self.sflv1_model_test()
        rmse = math.sqrt(mse)
        print(f"rmse: {rmse}, mape: {mape}, mae: {mae}")
    
    # SFLV2
    def sflv2_train(self):
        print('SFLV2 Training!')
        datas = [None] * len(self.clients)
        num_batches = len(self.clients[0].train_dataloader)
        self.sflv2_global_feature_extractor = feature_extractor(input_size=self.input_dim).to(device)
        self.sflv2_global_feature_processor = feature_processor().to(device)
        self.sflv2_global_regressor = regressor().to(device)

        # train
        for e in range(self.rounds):
            for id, client in enumerate(self.clients):
                datas[id] = iter(client.train_dataloader)
            for i in range(num_batches):
                for id, client in enumerate(self.clients):
                    X, Y = next(datas[id])
                    client.sflv2_train(X, Y)

                # Model average
                sfl_global_feature_processor_params = self.sflv2_processor_average() 
                self.sflv2_global_feature_processor.load_state_dict(sfl_global_feature_processor_params)

                for client in self.clients:
                    client.set_sflv2_feature_processor(sfl_global_feature_processor_params)

            # Model average
            sfl_global_feature_extractor_params, sfl_global_regressor_params = self.sflv2_extractor_regressor_average() 
            self.sflv2_global_feature_extractor.load_state_dict(sfl_global_feature_extractor_params)
            self.sflv2_global_regressor.load_state_dict(sfl_global_regressor_params)

            # Model distribute
            for client in self.clients:
                client.set_sflv2_feature_extractor(sfl_global_feature_extractor_params)
                client.set_sflv2_regressor(sfl_global_regressor_params)

        # Model finetune
        for i in range(30):
            for client in self.clients:
                client.sflv2_finetune()
                
        mse, mape, mae = self.sflv2_model_test()
        rmse = math.sqrt(mse)
        print(f"rmse: {rmse}, mape: {mape}, mae: {mae}")

    
    # Proposed
    def distillation_train(self):
        print('Proposed Training!')
        self.distillation_global_feature_extractor = feature_extractor(input_size=self.input_dim).to(device)
        self.distillation_global_feature_processor = feature_processor().to(device)
        self.distillation_global_regressor = regressor().to(device)
        self.distillation_global_auxiliary_regressor = regressor().to(device)

        # train
        for e in range(self.rounds):
            for client in self.clients:
                client.distillation_train(self.mu, self.gamma)

            # Model average
            distillation_global_feature_extractor_params, distillation_global_regressor_params, distillation_global_auxiliary_regressor_params = self.distillation_extractor_regressor_auxiliary_average()
            distillation_global_feature_processor_params = self.distillation_processor_average() 
            self.distillation_global_feature_processor.load_state_dict(distillation_global_feature_processor_params) 
            self.distillation_global_feature_extractor.load_state_dict(distillation_global_feature_extractor_params)
            self.distillation_global_regressor.load_state_dict(distillation_global_regressor_params)
            self.distillation_global_auxiliary_regressor.load_state_dict(distillation_global_auxiliary_regressor_params)

            # Model distribute
            for client in self.clients:
                client.set_distillation_feature_extractor(distillation_global_feature_extractor_params)
                client.set_distillation_regressor(distillation_global_regressor_params)
                client.set_distillation_auxiliary_regressor(distillation_global_auxiliary_regressor_params)
            for client in self.clients:
                client.set_distillation_feature_processor(distillation_global_feature_processor_params)
        
        # Model finetune
        for i in range(30):
            for client in self.clients:
                client.distillation_fine_tune()

        mse, mape, mae = self.distillation_model_test()
        rmse = math.sqrt(mse)
        print(f"rmse: {rmse}, mape: {mape}, mae: {mae}")

    