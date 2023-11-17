#ifndef __USART_H
#define __USART_H
#include "stdio.h"	
#include "stm32f4xx_conf.h"
#include "sys.h" 

#define EN_USART1_RX 			1	
	  	

void uart_init();
void Uart1_SendDATA(u8*SendBuf,int len);

void USART1_send_extractor_weight(float arr[]);
void USART1_send_extractor_bias(float arr[]);
void USART1_send_auxiliary_weight(float arr[]);
void USART1_send_auxiliary_bias(float arr[]);
void USART1_send_regressor_weight(float arr[]);
void USART1_send_regressor_bias(float arr[]);
void USART1_send_extractor_activation(float arr[]);
void USART1_send_processor_activation_gradient(float arr[]);
void USART1_send_extractor_output(float arr[]);
void USART1_receive_extractor_weight(void);
void USART1_receive_extractor_bias(void);
void USART1_receive_auxiliary_weight(void);
void USART1_receive_auxiliary_bias(void);
void USART1_receive_regressor_weight(void);
void USART1_receive_regressor_bias(void);
void USART1_receive_processor_activation(void);
void USART1_receive_extractor_activation_gradient(void);
void USART1_receive_processor_output(void);
void USART1_Receive(void);

#endif


