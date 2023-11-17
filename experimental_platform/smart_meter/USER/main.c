#include "delay.h"
#include "sys.h"
#include "led.h"
#include "lcd_init.h"
#include "lcd.h"
#include "pic.h"
#include "Neural Network.h"
#include "Dataset.h"
#include "tim.h"
#include "rng.h"
#include "stdio.h"
#include "math.h"
#include "usart.h"

int main(void)
{ 
	//************* Initialization **************//
	delay_init(168);
	LED_Init();
	LCD_Init();
	RNG_Init();
	uart_init();
	TIM7_Configuration();
	LCD_Fill(0,0,LCD_W,LCD_H,BLACK);		
	LED = ~LED;
	LCD_Initial();
	initial_weight();
	USART1_Receive();
	
	//************* Model Training in Federated Round **************//
	for(int i=0; i<Round; i++)
	{	
			LCD_round(i);
			for(int j=0; j<Batch_number; j++)
			{		
					read_training_data(j);
          extractor_forward();
					USART1_send_extractor_activation(Extractor_activation);
					USART1_receive_processor_activation();
					regressor_forward_backward();
					USART1_send_processor_activation_gradient(Processor_activation_gradient);
					auxiliary_forward_backward();
					extractor_backward();
					Adam_optimizer(i,j);
					LCD_batch(j);
			}
			
			//Model Average
			USART1_send_extractor_weight(Extractor_weight);	
			USART1_receive_extractor_weight();
			USART1_send_regressor_weight(Regression_weight);
			USART1_receive_regressor_weight();
			USART1_send_auxiliary_weight(Auxiliary_weight);
			USART1_receive_auxiliary_weight();
			USART1_send_extractor_bias(Extractor_bias);	
			USART1_receive_extractor_bias();
			USART1_send_regressor_bias(Regression_bias);
			USART1_receive_regressor_bias();
			USART1_send_auxiliary_bias(Auxiliary_bias);
			USART1_receive_auxiliary_bias();
      		
	}
	
	//************* Model Training in Fine-tuning Round **************//
	for(int i=0; i<Round_finetune; i++)
	{		
			LCD_round(i+Round);
			for(int j=0; j<Batch_number; j++)
			{		
					read_training_data(j);
          extractor_forward();
					USART1_send_extractor_activation(Extractor_activation);
					USART1_receive_processor_activation();
					regressor_forward_backward();
					USART1_send_processor_activation_gradient(Processor_activation_gradient);
					USART1_receive_extractor_activation_gradient();
					extractor_backward();
					Adam_optimizer(i+Round,j);
					LCD_batch(j);
			}
	}
	
	//************* Model Test **************//
	for(int k=0; k<Test_data_number-Input_timestep-Output_dimension; k++)
	{   
		  read_test_data(k);
			extractor_test();
			USART1_send_extractor_output(Extractor_activation_test);
			USART1_receive_processor_output();
			regressor_test();
	}
		
	while(1)
	{
	}
}
