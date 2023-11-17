#include "Neural Network.h"
#include "Dataset.h"
#include <math.h>
#include "stdlib.h"
#include "rng.h"
#include "lcd.h"

float Extractor_weight [Input_dimension * Hidden_dimension] = {0};
float Extractor_weight_gradient [Batch_size * Input_dimension * Hidden_dimension] = {0};
float Extractor_bias [Hidden_dimension] ={0};
float Extractor_bias_gradient [Batch_size * Hidden_dimension] ={0};
float Extractor_activation [Batch_size * Hidden_dimension] = {0};
float Extractor_activation_gradient [Batch_size * Hidden_dimension] = {0};
float Processor_activation [Batch_size * Hidden_dimension] = {0};
float Processor_activation_gradient [Batch_size * Hidden_dimension] = {0};
float Extractor_activation_test[Hidden_dimension] = {0};
float Processor_activation_test[Hidden_dimension] = {0};
float Regression_weight [Hidden_dimension * Output_dimension] = {0};
float Regression_weight_gradient [Batch_size * Hidden_dimension * Output_dimension] = {0};
float Regression_bias [Output_dimension] ={0};
float Regression_bias_gradient [Batch_size * Output_dimension] ={0};
float Auxiliary_weight [Hidden_dimension * Output_dimension] ={0};
float Auxiliary_weight_gradient [Batch_size * Hidden_dimension * Output_dimension] = {0};
float Auxiliary_bias [Output_dimension] = {0};
float Auxiliary_bias_gradient [Batch_size * Output_dimension] ={0};
float Input[Batch_size][Input_dimension] ={0};
float Label[Batch_size][Output_dimension] = {0};
float Input_single[Input_dimension] ={0};
float Label_single[Output_dimension] = {0};
float Prediction[Output_dimension] = {0};
float Prediction_auxiliary[Output_dimension] = {0};
float Prediction_inverse[Output_dimension] = {0};
float Label_inverse[Output_dimension] = {0};
float Extractor_bias_m [Hidden_dimension] = {0};
float Extractor_bias_v [Hidden_dimension] = {0};
float Extractor_weight_m[Input_dimension * Hidden_dimension] = {0};
float Extractor_weight_v[Input_dimension * Hidden_dimension] = {0};
float Auxiliary_bias_m [Output_dimension] = {0};
float Auxiliary_bias_v [Output_dimension] = {0};
float Auxiliary_weight_m[Hidden_dimension * Output_dimension] ={0};
float Auxiliary_weight_v[Hidden_dimension * Output_dimension] ={0};
float Regression_bias_m [Output_dimension] = {0};
float Regression_bias_v [Output_dimension] = {0};
float Regression_weight_m[Hidden_dimension * Output_dimension] ={0};
float Regression_weight_v[Hidden_dimension * Output_dimension] ={0};
float Prediction_show[Input_timestep]={0};
float True_show[Input_timestep]={0};
float mu = 0.5;
float gamma = 0.5;
float mse = 0;
float rmse = 0;
float mae = 0;
float mape = 0;


/******************************************************************
* Caculate model gradient with Adam optimizer											*
******************************************************************/
void Adam_optimizer(int a, int b)
{   
    int k = Batch_number*a+b+1;
		float m_hat = 0;
		float v_hat = 0;
		float o = 1.0;
			
		for (int i = 0; i < Output_dimension; i++)
		{
						for (int j = 0; j < Hidden_dimension; j++) 
						{											
								Auxiliary_weight_m[j*Output_dimension + i] = Beta1 * Auxiliary_weight_m[j*Output_dimension + i] + (o - Beta1) * Auxiliary_weight_gradient[j*Output_dimension + i];
								Auxiliary_weight_v[j*Output_dimension + i] = Beta2 * Auxiliary_weight_v[j*Output_dimension + i] + (o - Beta2) * Auxiliary_weight_gradient[j*Output_dimension + i] * Auxiliary_weight_gradient[j*Output_dimension + i];
								m_hat = Auxiliary_weight_m[j*Output_dimension+i]/(o - powf(Beta1, k));
								v_hat = Auxiliary_weight_v[j*Output_dimension + i] / (o - powf(Beta2, k));
								Auxiliary_weight[j*Output_dimension + i] = Auxiliary_weight[j*Output_dimension + i] - Learning_rate * m_hat / (sqrtf(v_hat) + Epsilon);						
								Auxiliary_weight_gradient[j*Output_dimension + i]=0;
						}
						Auxiliary_bias_m[i] = Beta1 * Auxiliary_bias_m[i] + (o - Beta1) * Auxiliary_bias_gradient[i];
						Auxiliary_bias_v[i] = Beta2 * Auxiliary_bias_v[i] + (o - Beta2) * Auxiliary_bias_gradient[i] * Auxiliary_bias_gradient[i];
						m_hat = Auxiliary_bias_m[i] / (o - powf(Beta1, k));
						v_hat = Auxiliary_bias_v[i] / (o - powf(Beta2, k));
						Auxiliary_bias[i] = Auxiliary_bias[i] - Learning_rate * m_hat / (sqrtf(v_hat) + Epsilon);
						Auxiliary_bias_gradient[i]=0;
						
		}
		
		for (int i = 0; i < Hidden_dimension; i++)
		{
						for (int j = 0; j < Input_dimension; j++) 
						{			
								
								Extractor_weight_m[j*Hidden_dimension+i] = Beta1 * Extractor_weight_m[j*Hidden_dimension+i] + (1 - Beta1) * Extractor_weight_gradient[j*Hidden_dimension+i];
								Extractor_weight_v[j*Hidden_dimension+i] = Beta2 * Extractor_weight_v[j*Hidden_dimension+i] + (1 - Beta2) * Extractor_weight_gradient[j*Hidden_dimension+i] * Extractor_weight_gradient[j*Hidden_dimension+i];
								m_hat =  Extractor_weight_m[j*Hidden_dimension+i] / (o - powf(Beta1, k));
								v_hat = Extractor_weight_v[j*Hidden_dimension+i] / (o - powf(Beta2, k));
								Extractor_weight[j*Hidden_dimension+i] = Extractor_weight[j*Hidden_dimension+i] - Learning_rate * m_hat / (sqrtf(v_hat) + Epsilon);					
								Extractor_weight_gradient[j*Hidden_dimension+i]=0;
								
						}
						Extractor_bias_m[i] = Beta1 * Extractor_bias_m[i] + (o - Beta1) * Extractor_bias_gradient[i];
						Extractor_bias_v[i] = Beta2 * Extractor_bias_v[i] + (o - Beta2) * Extractor_bias_gradient[i] * Extractor_bias_gradient[i];
						m_hat = Extractor_bias_m[i] / (o - powf(Beta1, k));
						v_hat = Extractor_bias_v[i] / (o - powf(Beta2, k));
						Extractor_bias[i] = Extractor_bias[i] - Learning_rate * m_hat / (sqrtf(v_hat) + Epsilon);
						Extractor_bias_gradient[i]=0;	
						
		}
		
		for (int i = 0; i < Output_dimension; i++)
		{
						for (int j = 0; j < Hidden_dimension; j++) 
						{											
								Regression_weight_m[j*Output_dimension + i] = Beta1 * Regression_weight_m[j*Output_dimension + i] + (o - Beta1) * Regression_weight_gradient[j*Output_dimension + i];
								Regression_weight_v[j*Output_dimension + i] = Beta2 * Regression_weight_v[j*Output_dimension + i] + (o - Beta2) * Regression_weight_gradient[j*Output_dimension + i] * Regression_weight_gradient[j*Output_dimension + i];
								m_hat = Regression_weight_m[j*Output_dimension+i]/(o - powf(Beta1, k));
								v_hat = Regression_weight_v[j*Output_dimension + i] / (o - powf(Beta2, k));
								Regression_weight[j*Output_dimension + i] = Regression_weight[j*Output_dimension + i] - Learning_rate * m_hat / (sqrtf(v_hat) + Epsilon);						
								Regression_weight_gradient[j*Output_dimension + i]=0;
						}
						Regression_bias_m[i] = Beta1 * Regression_bias_m[i] + (o - Beta1) * Regression_bias_gradient[i];
						Regression_bias_v[i] = Beta2 * Regression_bias_v[i] + (o - Beta2) * Regression_bias_gradient[i] * Regression_bias_gradient[i];
						m_hat = Regression_bias_m[i] / (o - powf(Beta1, k));
						v_hat = Regression_bias_v[i] / (o - powf(Beta2, k));
						Regression_bias[i] = Regression_bias[i] - Learning_rate * m_hat / (sqrtf(v_hat) + Epsilon);
						Regression_bias_gradient[i]=0;
						
		}
       
}


/******************************************************************
* Read training data from Flash to RAM
******************************************************************/

void read_training_data(int k)
{		
		for(int i=0; i<Batch_size; i++)
		{		 
		    for(int j=0; j<Input_timestep; j++)
				 {
							Input[i][j]=Trainy[k*Batch_size+i+j];						
		  		}
				 
				for(int m=0; m<Output_dimension; m++)
				{
						 for(int n=0; n<Calender_dimension; n++){
									Input[i][Input_timestep+m*Calender_dimension+n]=Trainx[k*Batch_size+i+Input_timestep+m][n];
						 }
						 Label[i][m]=Trainy[k*Batch_size + i+Input_timestep + m];
				}
	}
}

void read_testtrain_data(int k)
{		
				 
		    for(int i=0; i<Input_timestep; i++)
				{
							Input_single[i]=Trainy[k+i];						
		  	}
				 
				for(int j=0; j <Output_dimension;j++)
				{
						 for(int n=0; n<Calender_dimension;n++)
						 {
								Input_single[Input_timestep+j*Calender_dimension+n]=Trainx[k+Input_timestep+j][n];
						 }
						 Label_single[j]=Trainy[k+Input_timestep+j];
				}
	
}

/******************************************************************
* Read test data from Flash to RAM
******************************************************************/
void read_test_data(int k)
{		
				 
		    for(int i=0; i<Input_timestep; i++)
				{
							Input_single[i] = Testy[k+i];						
		  	}
				 
				for(int j=0; j <Output_dimension;j++)
				{
						 for(int n=0; n<Calender_dimension;n++)
						 {
								Input_single[Input_timestep+j*Calender_dimension+n]=Testx[k+Input_timestep+j][n];
						 }
						 Label_single[j]=Testy[k+Input_timestep+j];
				}
	
}
/******************************************************************
* Initialize model weights
******************************************************************/
void initial_weight(void) 
{
		float Rando;
		srand(123);
    for(int i=0; i<Hidden_dimension; i++) 
		{    
        for(int j=0; j <Input_dimension; j++)
				{ 
						Extractor_bias[i] = 0.0 ;
            Rando = (float)rand()/(float)RAND_MAX;
            Extractor_weight[j*Hidden_dimension + i] = (Rando - 0.5f)*Initial_max;
        }
    }

    for(int i=0; i<Output_dimension; i++) 
		{    
        for(int j=0; j < Hidden_dimension ;j++) 
				{
            Auxiliary_bias[i] = 0.0 ;  
            Rando = (float)rand()/(float)RAND_MAX;        
            Auxiliary_weight[j*Output_dimension + i] = (Rando - 0.5f)*Initial_max;
        }
    }
		
		for(int i=0; i<Output_dimension; i++) 
		{    
        for(int j=0; j < Hidden_dimension ;j++) 
				{
            Regression_bias[i] = 0.0 ;  
            Rando = (float)rand()/(float)RAND_MAX;        
            Regression_weight[j*Output_dimension + i] = (Rando - 0.5f)*Initial_max;
        }
    }

}

/******************************************************************
* Forward pass of extractor
******************************************************************/
void extractor_forward(void){
  float Accum =0;
	for (int k=0; k<Batch_size; k++)
	{
     for (int i=0; i<Hidden_dimension; i++)
		 {
        Accum = Extractor_bias[i];
        for (int j=0; j<Input_dimension; j++)
				{
            Accum += Input[k][j]* Extractor_weight[j*Hidden_dimension+ i];
        }
        Extractor_activation[k*Hidden_dimension+ i] = (Accum>0)?Accum:0;
      }
	 }
	
}

/******************************************************************
* Forward and backward pass of regressor
******************************************************************/
void regressor_forward_backward(void){

		float Loss[Output_dimension] = {0};
		float Accumf = 0.0;
		float Accumb = 0.0;
		
		for(int k=0; k<Batch_size; k++)
		{
				 //**Calculate prediction and loss
				 for(int i=0; i<Output_dimension; i++) 
				 {
						Accumf = Regression_bias[i];
						for(int j=0; j<Hidden_dimension; j++) 
						{
								Accumf += Processor_activation[k*Hidden_dimension + j] * Regression_weight[j*Output_dimension + i];
						}
						Prediction[i] = Accumf;
						Loss[i] = (Prediction[i] - Label[k][i])*0.5f;
						
					}
		
		
				 //**Calculate gradient	of regressor
				 for(int i=0; i<Output_dimension; i++ ) 
				 {    
						Regression_bias_gradient[i] += Loss[i] / Batch_size;
						for(int j=0; j < Hidden_dimension ; j++ ) 
						{
								Regression_weight_gradient[j*Output_dimension + i] += Processor_activation[k*Hidden_dimension+j] * Loss[i] / Batch_size;
						}
				}
				
				 //**Calculate activation gradient of processor
				 for(int i=0; i<Hidden_dimension; i++) 
				 {    
							Accumb = 0.0;
							for(int j=0; j<Output_dimension; j++) 
							{
									Accumb += Regression_weight[i*Output_dimension+j] * Loss[j];
							}
							Processor_activation_gradient[k*Hidden_dimension+i] = Accumb;
					}
		}
}

/******************************************************************
* Forward and backward pass of auxiliary regressor
******************************************************************/
void auxiliary_forward_backward(void){

		float Loss[Output_dimension] = {0};
		float Accumf = 0.0;
		float Accumb = 0.0;
		
		for(int k=0; k<Batch_size; k++)
		{
				 //**Calculate prediction and loss
				 for(int i=0; i<Output_dimension; i++) 
				 {
						Accumf = Auxiliary_bias[i];
						for(int j=0; j<Hidden_dimension; j++) 
						{
								Accumf += Extractor_activation[k*Hidden_dimension + j] * Auxiliary_weight[j*Output_dimension + i];
						}
						
						Prediction_auxiliary[i] = Accumf;
						Loss[i] = mu*(Prediction_auxiliary[i] - Label[k][i])*0.5f + gamma*(Prediction_auxiliary[i] - Prediction[i])*0.5f;
					}
		
		
				 //**Calculate gradient	of auxiliary regressor
				 for(int i=0 ; i < Output_dimension; i++ ) 
				 {    
						Auxiliary_bias_gradient[i] += Loss[i] / Batch_size ;
						for(int j=0 ; j < Hidden_dimension ; j++ ) 
						{
								Auxiliary_weight_gradient[j*Output_dimension + i] += Extractor_activation[k*Hidden_dimension+j] * Loss[i] / Batch_size;
						}
				}
				
				 //**Calculate activation gradient of extractor
				 for(int i=0; i<Hidden_dimension; i++) 
				 {    
							Accumb = 0.0;
							for(int j=0; j<Output_dimension; j++) 
							{
									Accumb += Auxiliary_weight[i*Output_dimension+j] * Loss[j];
							}
							
							Extractor_activation_gradient[k*Hidden_dimension+i] = Accumb;
					}
		}
}

/******************************************************************
* Backward pass of extractor in federated round
******************************************************************/
void extractor_backward(void){
		
		float HiddenDelta [Hidden_dimension] = {0};
		for(int k=0; k<Batch_size; k++)
		{				 
				 //**Calculate gradient of extractor
				 for(int i=0 ; i<Hidden_dimension; i++ ) 
				 {   
						HiddenDelta[i] = (Extractor_activation[k*Hidden_dimension+i]>0)?Extractor_activation_gradient[k*Hidden_dimension+i]:0;
						Extractor_bias_gradient[i] += HiddenDelta[i]/Batch_size;
						for(int j=0; j<Input_dimension; j++)
						{ 
								Extractor_weight_gradient[j*Hidden_dimension+i] += Input[k][j] * HiddenDelta[i]/Batch_size;           
						}
				}
				
		}
}

/******************************************************************
* Backward pass of extractor in fine-tuning round
******************************************************************/
void extractor_backward_finetune(void){
		
		float HiddenDelta [Hidden_dimension] = {0};
		for(int k=0; k<Batch_size; k++)
		{				 
				 //**Calculate gradient of extractor
				 for(int i=0 ; i<Hidden_dimension; i++ ) 
				 {   
						HiddenDelta[i] = (Extractor_activation[k*Hidden_dimension+i]>0)?Extractor_activation_gradient[k*Hidden_dimension+i]:0;
						Extractor_bias_gradient[i] += HiddenDelta[i]/Batch_size;
						for(int j=0; j<Input_dimension; j++)
						{ 
								Extractor_weight_gradient[j*Hidden_dimension+i] += Input[k][j] * HiddenDelta[i]/Batch_size;           
						}
				}
				
		}
}

/******************************************************************
* Model test of extractor
******************************************************************/
void extractor_test(void)
{
    float Accum=0;
		for (int i = 0; i < Hidden_dimension; i++) 
		{
					Accum = Extractor_bias[i];
					for (int j = 0; j < Input_dimension; j++) 
					{
							Accum += Input_single[j] * Extractor_weight[j*Hidden_dimension + i];
					}
					Extractor_activation_test[i] = (Accum>0)?Accum:0;
		}
}					 
	
/******************************************************************
* Model test of regressor
******************************************************************/
void regressor_test(void)
{
		float Accumf=0;
		for (int i = 0; i < Output_dimension; i++) 
		{
					Accumf = Regression_bias[i];
					for (int j = 0; j < Hidden_dimension; j++) 
					{
							Accumf += Processor_activation_test[j] * Regression_weight[j*Output_dimension + i];
					}
					Prediction[i] = Accumf;									
		}

		
		for (int i = 0; i < Output_dimension; i++) 
		{	
					Prediction_inverse[i] = Var*Prediction[i] + Mean;
				  Label_inverse[i] = Var*Label_single[i] + Mean;
					mse += (Prediction_inverse[i] - Label_inverse[i])*(Prediction_inverse[i] - Label_inverse[i]);	
		}								 
}


