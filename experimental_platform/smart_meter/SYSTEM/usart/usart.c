#include "sys.h"
#include "usart.h"	
#include "Neural Network.h"
#include "dataset.h"
#include "math.h"
////////////////////////////////////////////////////////////////////////////////// 	 

 

   
//////////////////////////////////////////////////////////////////
              
struct __FILE 
{ 
	int handle; 
}; 

FILE __stdout;       
   
void _sys_exit(int x) 
{ 
	x = x; 
} 
 
int fputc(int ch, FILE *f)
{ 	
	while((USART1->SR&0X40)==0);   
	USART1->DR = (u8) ch;      
	return ch;
}

void USART_Send(u8 data)
{
    while ((USART1->SR&0X40)==0);
    USART1->DR = data;
}

void Uart1_SendDATA(u8*SendBuf,int len)
{
   int i;
    for(i=0;i<len;i++)
	{
	  while((USART1->SR&0X40)==0); 
      USART1->DR = (u8) SendBuf[i]; 
	}
}

void USART1_SendFloat(float value) {
    u8 * p = (char *)&value; 
    for (int i = 0; i < sizeof(float); i++) 
		{
        USART_Send(p[i]); 
    }
}

u8 USART1_Read(void) {
    while (!(USART1->SR & USART_SR_RXNE)){
		}
    return USART1->DR;
}


void USART1_send_extractor_weight(float arr[])
{			
		USART_Send(Identifier_extractor_weight);
    for (int i = 0; i < Hidden_dimension; i++)
    {
        for (int j = 0; j < Input_dimension; j++)
        {
            float value = arr[i*Input_dimension+j];
            u8 *value_bytes = (char *)&value;
            for (int k = 0; k < sizeof(float); k++)
            {
                USART_Send(value_bytes[k]);
            }
        }
    }
}

void USART1_send_extractor_bias(float arr[])
{			
		USART_Send(Identifier_extractor_bias);
    for (int i = 0; i < Hidden_dimension; i++)
    {
            float value = arr[i];
            u8 *value_bytes = (char *)&value;
            for (int k = 0; k < sizeof(float); k++)
            {
                USART_Send(value_bytes[k]);
            }
        
    }
}

void USART1_send_auxiliary_weight(float arr[])
{			
		USART_Send(Identifier_auxiliary_weight);
    for (int i = 0; i < Output_dimension; i++)
    {
        for (int j = 0; j < Hidden_dimension; j++)
        {
            float value = arr[i*Hidden_dimension+j];
            u8 *value_bytes = (char *)&value;
            for (int k = 0; k < sizeof(float); k++)
            {
                USART_Send(value_bytes[k]);
            }
        }
    }
}

void USART1_send_auxiliary_bias(float arr[])
{			
		USART_Send(Identifier_auxiliary_bias);
    for (int i = 0; i < Output_dimension; i++)
    {
            float value = arr[i];
            u8 *value_bytes = (char *)&value;
            for (int k = 0; k < sizeof(float); k++)
            {
                USART_Send(value_bytes[k]);
            }
    }
}

void USART1_send_regressor_weight(float arr[])
{			
		USART_Send(Identifier_regressor_weight);
    for (int i = 0; i < Output_dimension; i++)
    {
        for (int j = 0; j < Hidden_dimension; j++)
        {
            float value = arr[i*Hidden_dimension+j];
            u8 *value_bytes = (char *)&value;
            for (int k = 0; k < sizeof(float); k++)
            {
                USART_Send(value_bytes[k]);
            }
        }
    }
}

void USART1_send_regressor_bias(float arr[])
{			
		USART_Send(Identifier_regressor_bias);
    for (int i = 0; i < Output_dimension; i++)
    {
            float value = arr[i];
            u8 *value_bytes = (char *)&value;
            for (int k = 0; k < sizeof(float); k++)
            {
                USART_Send(value_bytes[k]);
            }
        
    }
}

void USART1_send_extractor_activation(float arr[])
{			
		USART_Send(Identifier_extractor_activation);
    for (int i = 0; i < Batch_size; i++)
    {
        for (int j = 0; j < Hidden_dimension; j++)
        {
            float value = arr[i*Hidden_dimension+j];
            u8 *value_bytes = (char *)&value;
            for (int k = 0; k < sizeof(float); k++)
            {
                USART_Send(value_bytes[k]);
            }
        }
    }
}

void USART1_send_processor_activation_gradient(float arr[])
{			
		USART_Send(Identifier_processor_activation_gradient);
    for (int i = 0; i < Batch_size; i++)
    {
        for (int j = 0; j < Hidden_dimension; j++)
        {
            float value = arr[i*Hidden_dimension+j];
            u8 *value_bytes = (char *)&value;
            for (int k = 0; k < sizeof(float); k++)
            {
                USART_Send(value_bytes[k]);
            }
        }
    }
}

void USART1_send_extractor_output(float arr[])
{			
		USART_Send(Identifier_extractor_output);
    for (int i = 0; i < Hidden_dimension; i++)
    {
        float value = arr[i];
        u8 *value_bytes = (char *)&value;
        for (int k = 0; k < sizeof(float); k++)
        {
            USART_Send(value_bytes[k]);
        }
        
    }
}


void USART1_receive_extractor_weight(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Input_dimension*Hidden_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Extractor_weight[i], buffer, sizeof(float));
		}
}

void USART1_receive_extractor_bias(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Hidden_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Extractor_weight_gradient[i], buffer, sizeof(float));
		}
}

void USART1_receive_auxiliary_weight(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Output_dimension*Hidden_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Auxiliary_weight[i], buffer, sizeof(float));
		}

}

void USART1_receive_auxiliary_bias(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Output_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Auxiliary_bias[i], buffer, sizeof(float));
		}

}

void USART1_receive_regressor_weight(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Output_dimension*Hidden_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Regression_weight[i], buffer, sizeof(float));
		}

}

void USART1_receive_regressor_bias(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Output_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Regression_bias[i], buffer, sizeof(float));
		}

}

void USART1_receive_processor_activation(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Batch_size*Hidden_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Processor_activation[i], buffer, sizeof(float));
		}

}

void USART1_receive_extractor_activation_gradient(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Batch_size*Hidden_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Extractor_activation_gradient[i], buffer, sizeof(float));
		}

}

void USART1_receive_processor_output(void) 
{
    u8 Res;
		int bytecount = 0;
		u8 buffer[sizeof(float)];
		for (int i = 0; i < Hidden_dimension; i++)
    {
        for(int j=0; j<sizeof(float);j++)
				{
						Res=USART1_Read();
						buffer[j]=Res;
				}
				memcpy(&Processor_activation_test[i], buffer, sizeof(float));
		}

}

void USART1_Receive(void) 
{
    u8 Res;
		Res=USART1_Read();
			
}



void uart_init(){

  GPIO_InitTypeDef GPIO_InitStructure;
	USART_InitTypeDef USART_InitStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
	
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA,ENABLE); 
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1,ENABLE);
 
	GPIO_PinAFConfig(GPIOA,GPIO_PinSource9,GPIO_AF_USART1); 
	GPIO_PinAFConfig(GPIOA,GPIO_PinSource10,GPIO_AF_USART1); 
	
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9 | GPIO_Pin_10; 
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;	
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP; 
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP; 
	GPIO_Init(GPIOA,&GPIO_InitStructure); 

  
	USART_InitStructure.USART_BaudRate = 115200;//Set baud rate
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
	USART_InitStructure.USART_StopBits = USART_StopBits_1;
	USART_InitStructure.USART_Parity = USART_Parity_No;
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
  USART_Init(USART1, &USART_InitStructure); 
	
  USART_Cmd(USART1, ENABLE);
	
	USART_ClearFlag(USART1, USART_FLAG_TC);
	
	USART_ITConfig(USART1, USART_IT_RXNE, DISABLE);
}
 


 



