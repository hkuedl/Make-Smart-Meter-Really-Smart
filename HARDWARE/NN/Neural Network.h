#ifndef __NN_H
#define __NN_H 			
#include "sys.h"

#define Beta1    						0.9f    //Parameters for Adam optimizer
#define Beta2    						0.999f
#define Epsilon   					1e-8f
#define Initial_max  				0.5f
#define Learning_rate 			0.0005f
#define Round        				100
#define Round_finetune   		30
#define Batch_size    			32
#define Calender_dimension  4
#define Input_timestep      24
#define Input_dimension   	40
#define Hidden_dimension  	16
#define Output_dimension  	4
#define Batch_number				436
#define Test_data_number    3516

#define ARM_MATH_CM4

void Adam_optimizer(int a, int b);
void read_training_data(int k);
void read_test_data(int k);
void initial_weight(void);
void extractor_forward(void);
void regressor_forward_backward(void);
void auxiliary_forward_backward(void);
void extractor_backward(void);	
void regressor_test(void);
void extractor_test(void);


extern float Extractor_weight [Input_dimension * Hidden_dimension];
extern float Extractor_weight_gradient [Batch_size * Input_dimension * Hidden_dimension];
extern float Extractor_bias [Hidden_dimension];
extern float Extractor_bias_gradient [Batch_size * Hidden_dimension];
extern float Extractor_activation [Batch_size * Hidden_dimension];
extern float Extractor_activation_gradient [Batch_size * Hidden_dimension];
extern float Processor_activation [Batch_size * Hidden_dimension];
extern float Processor_activation_gradient [Batch_size * Hidden_dimension];
extern float Extractor_activation_test[Hidden_dimension];
extern float Processor_activation_test[Hidden_dimension];
extern float Regression_weight [Hidden_dimension * Output_dimension];
extern float Regression_weight_gradient [Batch_size * Hidden_dimension * Output_dimension];
extern float Regression_bias [Output_dimension];
extern float Regression_bias_gradient [Batch_size * Output_dimension];
extern float Auxiliary_weight [Hidden_dimension * Output_dimension];
extern float Auxiliary_weight_gradient [Batch_size * Hidden_dimension * Output_dimension];
extern float Auxiliary_bias [Output_dimension];
extern float Auxiliary_bias_gradient [Batch_size * Output_dimension];
extern float Input[Batch_size][Input_dimension];
extern float Label[Batch_size][Output_dimension];
extern float Input_single[Input_dimension];
extern float Label_single[Output_dimension];
extern float Prediction[Output_dimension];
extern float Prediction_inverse[Output_dimension];
extern float Label_inverse[Output_dimension];
extern float Prediction_show[Input_timestep];
extern float True_show[Input_timestep];
extern float mse;
extern float rmse;
extern float mae;
extern float mape;


#endif
