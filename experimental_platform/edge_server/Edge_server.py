import serial
import struct
import numpy as np
import json
import socket
import torch
from Trainer_edge import edge_server, setup_seed
from multiprocessing import Process, Queue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_float_array(input, output, port):
    arr = [[0.0] * input for _ in range(output)]
    for i in range(output):
        for j in range(input):
            data = port.read(4)
            if len(data)==4:
                value = struct.unpack('<f', data)[0]
                arr[i][j] = value
    return arr

def read_float_bias(input, port):
    arr = [0.0] * input
    for i in range(input):
        data = port.read(4)
        if len(data)==4:
            value = struct.unpack('<f', data)[0]
            arr[i]= value
    return arr

#============================================================================
#     Receive the parameters from smart meter and upload to cloud server
#============================================================================
def read_data(ports, queue, queue_p, server_ip, server_port):
    inputnodes = 40
    hiddennodes = 32
    outputnodes = 4
    batch = 32
    server_complete = 0
    float_extractorweight = np.zeros((hiddennodes, inputnodes)).astype(np.float32)
    float_auxiliaryweight = np.zeros((outputnodes, hiddennodes)).astype(np.float32)
    float_regressorweight = np.zeros((outputnodes, hiddennodes)).astype(np.float32)
    float_extractorbias = np.zeros((hiddennodes)).astype(np.float32)
    float_auxiliarybias = np.zeros((outputnodes)).astype(np.float32)
    float_regressorbias = np.zeros((outputnodes)).astype(np.float32)
    extractor_grad = np.zeros((batch, hiddennodes)).astype(np.float32)
    extractor_act = np.zeros((batch, hiddennodes)).astype(np.float32)
    server_act = np.zeros((batch, hiddennodes)).astype(np.float32)
    extractor_act_test = np.zeros((hiddennodes)).astype(np.float32)
    client = edge_server()
    
    edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    edge_socket.connect((server_ip, server_port))
    
    ser = serial.Serial(port=ports, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, timeout=1)
    ser.write(0x01)
    
    # deliver the parameters from smart meter to process_data via queue
    while True:
        if ser.in_waiting:
            identifier = ser.read(1)
            if ord(identifier) & 0xF0 == 0x10: 
                arr_extractor = read_float_array(inputnodes, hiddennodes, ser)
                float_extractorweight[:,:]  = np.array(arr_extractor, dtype=np.float32)     
                queue.put((1, float_extractorweight))
                    
            elif ord(identifier) & 0xF0 == 0x20:
                arr_auxiliary = read_float_array(hiddennodes, outputnodes, ser)
                float_auxiliaryweight[:,:]  = np.array(arr_auxiliary, dtype=np.float32)      
                queue.put((2, float_auxiliaryweight))
                
            elif ord(identifier) & 0xF0 == 0x30:
                arrregressor = read_float_array(hiddennodes, outputnodes, ser)
                float_regressorweight[:,:]  = np.array(arrregressor, dtype=np.float32)          
                queue.put((3, float_regressorweight))
                        
            elif ord(identifier) & 0xF0 == 0x40: 
                arr_extractor = read_float_bias(hiddennodes,ser)
                float_extractorbias[:] = np.array(arr_extractor, dtype=np.float32)     
                queue.put((4, float_extractorbias))
                        
            elif ord(identifier) & 0xF0 == 0x50:
                arr_auxiliary = read_float_bias(outputnodes, ser)
                float_auxiliarybias[:] = np.array(arr_auxiliary, dtype=np.float32)        
                queue.put((5, float_auxiliarybias))
                        
            elif ord(identifier) & 0xF0 == 0x60:
                arrregressor = read_float_bias(outputnodes, ser)
                float_regressorbias[:] = np.array(arrregressor, dtype=np.float32)       
                queue.put((6, float_regressorbias))
                                   
            elif ord(identifier) & 0xF0 == 0x70:
                res = read_float_array(hiddennodes, batch, ser)
                extractor_act[:,:] = np.array(res, dtype=np.float32)
                extractor_act_tensor = torch.tensor(extractor_act[:,:].reshape(32,32)).to(device)
                server_act[:,:] = client.forward(extractor_act_tensor)
                byte_act = bytearray()
                extractor_server_act = server_act[:,:]
                for row_server in extractor_server_act:
                    for value_server in row_server:
                        byte_act.extend(struct.pack("f", value_server))
                ser.write(byte_act)
                
            elif ord(identifier) & 0xF0 == 0x80:
                res = read_float_array(hiddennodes, batch, ser)
                extractor_grad[:,:] = np.array(res, dtype=np.float32)
                extractor_grad_tensor = torch.tensor(extractor_grad[:,:].reshape(32,32)).to(device)
                extractor_act_tensor = torch.tensor(extractor_act[:,:].reshape(32,32), requires_grad=True).to(device)
                extractor_grad[:,:] = client.trainer(extractor_grad_tensor, extractor_act_tensor)
                model_client = client.get_model()
                queue.put((9, model_client))   
                                
            elif ord(identifier) & 0xF0 == 0x90:
                act_test = read_float_bias(hiddennodes, ser)
                extractor_act_test[:] = np.array(act_test, dtype=np.float32)
                extractor_act_tensor_test = torch.tensor(extractor_act_test[:].reshape(32)).to(device)
                extractor_act_test[:] = client.forward(extractor_act_tensor_test)
                extractor_server_act_test = extractor_act_test[:]
                byte_act_test = bytearray()
                for value_server in extractor_server_act_test:
                    byte_act_test.extend(struct.pack("f", value_server))               
                ser.write(byte_act_test)
        
        # receive the processed parameters via queue        
        if not queue_p.empty():
            id, data = queue_p.get()
            if id == 1:
                byte_data_extractor = bytearray()
                data_to_send = {
                        "client_id": id,
                        "neural_net_params": data
                }
                data_json = json.dumps(data_to_send)
                edge_socket.sendall(data_json.encode())
                response = edge_socket.recv(1024)
                for row_extractor in response:
                    for value_extractor in row_extractor :
                        byte_data_extractor.extend(struct.pack("f", value_extractor))
                ser.write(byte_data_extractor)
                
            elif id == 2:
                byte_data_auxiliary = bytearray()
                data_to_send = {
                        "client_id": id,
                        "neural_net_params": data
                }
                data_json = json.dumps(data_to_send)
                edge_socket.sendall(data_json.encode())
                response = edge_socket.recv(1024)
                for row_auxiliary in response:
                    for value_auxiliary in row_auxiliary:
                        byte_data_auxiliary.extend(struct.pack("f", value_auxiliary)) 
                ser.write(byte_data_auxiliary)
                
            elif id == 3:
                byte_data_regressor = bytearray()
                data_to_send = {
                        "client_id": id,
                        "neural_net_params": data
                }
                data_json = json.dumps(data_to_send)
                edge_socket.sendall(data_json.encode())
                response = edge_socket.recv(1024)
                for rowregressor in response:
                    for valueregressor in rowregressor:
                        byte_data_regressor.extend(struct.pack("f", valueregressor))
                ser.write(byte_data_regressor)
                
            elif id == 4:
                byte_data_extractor = bytearray()
                data_to_send = {
                        "client_id": id,
                        "neural_net_params": data
                }
                data_json = json.dumps(data_to_send)
                edge_socket.sendall(data_json.encode())
                response = edge_socket.recv(1024)
                for value_extractor in response:
                    byte_data_extractor.extend(struct.pack("f", value_extractor))
                ser.write(byte_data_extractor)
                        
            elif id == 5:
                byte_data_auxiliary = bytearray()
                data_to_send = {
                        "client_id": id,
                        "neural_net_params": data
                }
                data_json = json.dumps(data_to_send)
                edge_socket.sendall(data_json.encode())
                response = edge_socket.recv(1024)
                for value_auxiliary in response:
                    byte_data_auxiliary.extend(struct.pack("f", value_auxiliary))
                ser.write(byte_data_auxiliary)
            
            elif id == 6:
                byte_data_regressor = bytearray()
                data_to_send = {
                        "client_id": id,
                        "neural_net_params": data
                }
                data_json = json.dumps(data_to_send)
                edge_socket.sendall(data_json.encode())
                response = edge_socket.recv(1024)
                for valueregressor in response:
                    byte_data_regressor.extend(struct.pack("f", valueregressor))
                ser.write(byte_data_regressor)
            
            elif id == 9:    
                byte_data_regressor = bytearray()
                data_to_send = {
                        "client_id": id,
                        "neural_net_params": data
                }
                data_json = json.dumps(data_to_send)
                edge_socket.sendall(data_json.encode())
                response = edge_socket.recv(1024)
                client.set_model(response)
                server_complete = 1
                
        if server_complete == 1:
            server_complete = 0
            byte_grad = bytearray()
            extractor_server_grad = extractor_grad[:,:]    
            for row_server in extractor_server_grad:
                for value_server in row_server:
                    byte_grad.extend(struct.pack("f", value_server))
            ser.write(byte_grad) 


#===============================================================
#     Process the parameters from smart meter
#===============================================================   
def process_data(queue_s, queue_p):
    extractorweight =[]
    auxiliaryweight = []
    regressorweight = []
    extractorbias = []
    auxiliarybias = []
    regressorbias = []
    serverweight = []
    while True:
        for queue in queue_s:
            if not queue.empty():
                id, data = queue.get()
                if id == 1:
                    extractorweight.append(data)
                    if len(extractorweight) == 10:
                        arr = np.array(extractorweight)
                        avg_extractorweight = arr.mean(axis=0)
                        avg_extractorweight = avg_extractorweight.astype(np.float32)
                        has_nan = np.isnan(avg_extractorweight).any()
                        if has_nan:
                            print("avg_extractorweight")
                        extractorweight.clear()
                        for qut in queue_p:
                            qut.put((1, avg_extractorweight))
                        
                elif id == 2:
                    auxiliaryweight.append(data)
                    if len(auxiliaryweight) == 10:
                        arr = np.array(auxiliaryweight)
                        avg_auxiliaryweight = arr.mean(axis=0)
                        avg_auxiliaryweight = avg_auxiliaryweight.astype(np.float32)
                        auxiliaryweight.clear()
                        for qut in queue_p:
                            qut.put((2, avg_auxiliaryweight))        
                    
                elif id == 3:
                    regressorweight.append(data)
                    if len(regressorweight) == 10:
                        arr = np.array(regressorweight)
                        avg_regressorweight = arr.mean(axis=0)
                        avg_regressorweight = avg_regressorweight.astype(np.float32)
                        has_nan = np.isnan(avg_regressorweight).any()
                        if has_nan:
                            print("avg_regressorweight")
                        regressorweight.clear()
                        for qut in queue_p:
                            qut.put((3, avg_regressorweight))
                            
                elif id == 4:
                    extractorbias.append(data)
                    if len(extractorbias) == 10:
                        arr = np.array(extractorbias)
                        avg_extractorbias = arr.mean(axis=0)
                        avg_extractorbias = avg_extractorbias.astype(np.float32)
                        has_nan = np.isnan(avg_extractorbias).any()
                        if has_nan:
                            print("avg_extractorbias")
                        extractorbias.clear()
                        for qut in queue_p:
                            qut.put((4, avg_extractorbias))
                    
                elif id == 5:
                    auxiliarybias.append(data)
                    if len(auxiliarybias) == 10:
                        arr = np.array(auxiliarybias)
                        avg_auxiliarybias = arr.mean(axis=0)
                        avg_auxiliarybias = avg_auxiliarybias.astype(np.float32)
                        auxiliarybias.clear()
                        for qut in queue_p:
                            qut.put((5, avg_auxiliarybias))
                            
                elif id == 6:
                    regressorbias.append(data)
                    if len(regressorbias) == 10:
                        arr = np.array(regressorbias)
                        avg_regressorbias = np.mean(arr, axis=0)
                        avg_regressorbias = avg_regressorbias.astype(np.float32)
                        has_nan = np.isnan(avg_regressorbias).any()
                        if has_nan:
                            print("avg_regressorbias")
                        regressorbias.clear()
                        for qut in queue_p:
                            qut.put((6, avg_regressorbias))                        
                            
                elif id == 9:
                    serverweight.append(data)
                    if len(serverweight) == 10:
                        avg_params = serverweight[0]
                        for param_name in avg_params:
                            for i in range(1, len(serverweight)):
                                keys = serverweight[i]
                                avg_params[param_name] += keys[param_name]
                            avg_params[param_name] /= len(serverweight)
                        for queue in queue_p:
                            queue.put((9, avg_params))
                        serverweight.clear()
                      
if __name__ == '__main__':
    queue_s = []
    queue_p = []
    paraller = []
    num_workers = 10
    setup_seed(99)
    # Smart meter serial port
    ser_port = ["COM{}".format(i) for i in range(3, 33)]
    # Cloud server ip and port
    server_ip = 'xxx.xxx.x.xxx'
    server_port = 8080
    
    for _ in range(10):
        que = Queue()
        queue_s.append(que)
        quet = Queue()
        queue_p.append(quet)
          
    for i in range(10):
        parl = Process(target = read_data, args=(ser_port[i], queue_s[i], queue_p[i], server_ip, server_port))
        parl.start() 
        paraller.append(parl)
            
    process = Process(target = process_data, args =(queue_s, queue_p))
    process.start()    
        
    for p in paraller:
        p.join()

    process.join()  

                                                 