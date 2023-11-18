import socket
import json
import torch
from Trainer_cloud import feature_extractor, feature_processor, regressor, aggregate_params

alpha = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Cloud server ip and port
    server_ip = 'xxx.xxx.x.xxx'
    server_port = 8080

    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)

    global_feature_extractor = feature_extractor().to(device)
    global_feature_processor = feature_processor().to(device)
    global_regressor = regressor().to(device)
    global_auxiliary_regressor = regressor().to(device)
    print("Server is listening on {}:{}".format(server_ip, server_port))

    while True:
        # Waiting for the client to connect
        client_socket, client_address = server_socket.accept()
        print("Client connected:", client_address)

        try:
            while True:
                # Receive data sent by the client
                data = client_socket.recv(1024)
                received_data = json.loads(data.decode())
                
                 # Aggregate neural network parameters
                parameter_id = received_data["client_id"]
                neural_net_params = received_data["neural_net_params"]

                if parameter_id == 1:
                    aggregated_params = aggregate_params(global_feature_extractor, neural_net_params, alpha)
                
                elif parameter_id == 2:
                    aggregated_params = aggregate_params(global_auxiliary_regressor, neural_net_params, alpha)
                    
                elif parameter_id == 3:
                    aggregated_params = aggregate_params(global_regressor, neural_net_params, alpha)
                    
                elif parameter_id == 4:
                    aggregated_params = aggregate_params(global_feature_extractor, neural_net_params, alpha)
                    
                elif parameter_id == 5:
                    aggregated_params = aggregate_params(global_auxiliary_regressor, neural_net_params, alpha)
                
                elif parameter_id == 6:
                    aggregated_params = aggregate_params(global_regressor, neural_net_params, alpha)
                    
                elif parameter_id == 9:
                    aggregated_params = aggregate_params(global_feature_processor, neural_net_params, alpha)    
                
                response_json = json.dumps(aggregated_params)
                client_socket.sendall(response_json.encode())

        except Exception as e:
            print("Error:", e)

        finally:
            client_socket.close()