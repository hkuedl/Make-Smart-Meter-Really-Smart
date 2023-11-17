#include "lcd.h"
#include "lcd_init.h"
#include "lcdfont.h"
#include "delay.h"
#include "stdio.h"
#include "Neural Network.h"
#include "Dataset.h"
#include "tim.h"
#include "usart.h"

void LCD_Fill(u16 xsta, u16 ysta, u16 xend, u16 yend, u16 color)
{
	u16 i, j;
	LCD_Address_Set(xsta, ysta, xend - 1, yend - 1);
	for(i = ysta; i < yend; i++)
	{
		for(j = xsta; j < xend; j++)
		{
			LCD_WR_DATA(color);
		}
	}
}

void LCD_DrawPoint(u16 x, u16 y, u16 color)
{
	LCD_Address_Set(x, y, x, y); 
	LCD_WR_DATA(color);
}


void LCD_DrawLine(u16 x1, u16 y1, u16 x2, u16 y2, u16 color)
{
	u16 t;
	int xerr = 0, yerr = 0, delta_x, delta_y, distance;
	int incx, incy, uRow, uCol;
	delta_x = x2 - x1; 
	delta_y = y2 - y1;
	uRow = x1;
	uCol = y1;
	if(delta_x > 0)incx = 1;
	else if (delta_x == 0)incx = 0;
	else
	{
		incx = -1;
		delta_x = -delta_x;
	}
	if(delta_y > 0)incy = 1;
	else if (delta_y == 0)incy = 0;
	else
	{
		incy = -1;
		delta_y = -delta_y;
	}
	if(delta_x > delta_y)distance = delta_x; 
	else distance = delta_y;
	for(t = 0; t < distance + 1; t++)
	{
		LCD_DrawPoint(uRow, uCol, color); 
		xerr += delta_x;
		yerr += delta_y;
		if(xerr > distance)
		{
			xerr -= distance;
			uRow += incx;
		}
		if(yerr > distance)
		{
			yerr -= distance;
			uCol += incy;
		}
	}
}


void LCD_DrawRectangle(u16 x1, u16 y1, u16 x2, u16 y2, u16 color)
{
	LCD_DrawLine(x1, y1, x2, y1, color);
	LCD_DrawLine(x1, y1, x1, y2, color);
	LCD_DrawLine(x1, y2, x2, y2, color);
	LCD_DrawLine(x2, y1, x2, y2, color);
}

void Draw_Circle(u16 x0, u16 y0, u8 r, u16 color)
{
	int a, b;
	a = 0;
	b = r;
	while(a <= b)
	{
		LCD_DrawPoint(x0 - b, y0 - a, color);       //3
		LCD_DrawPoint(x0 + b, y0 - a, color);       //0
		LCD_DrawPoint(x0 - a, y0 + b, color);       //1
		LCD_DrawPoint(x0 - a, y0 - b, color);       //2
		LCD_DrawPoint(x0 + b, y0 + a, color);       //4
		LCD_DrawPoint(x0 + a, y0 - b, color);       //5
		LCD_DrawPoint(x0 + a, y0 + b, color);       //6
		LCD_DrawPoint(x0 - b, y0 + a, color);       //7
		a++;
		if((a * a + b * b) > (r * r))
		{
			b--;
		}
	}
}

void LCD_ShowChar(u16 x, u16 y, u8 num, u16 fc, u16 bc, u8 sizey, u8 mode)
{
	u8 temp, sizex, t, m = 0;
	u16 i, TypefaceNum;
	u16 x0 = x;
	sizex = sizey / 2;
	TypefaceNum = (sizex / 8 + ((sizex % 8) ? 1 : 0)) * sizey;
	num = num - ' ';
	LCD_Address_Set(x, y, x + sizex - 1, y + sizey - 1);
	for(i = 0; i < TypefaceNum; i++)
	{
		if(sizey == 12)temp = ascii_1206[num][i];
		else if(sizey == 16)temp = ascii_1608[num][i];
		else if(sizey == 24)temp = ascii_2412[num][i];
		else if(sizey == 32)temp = ascii_3216[num][i];
		else return;
		for(t = 0; t < 8; t++)
		{
			if(!mode)
			{
				if(temp & (0x01 << t))LCD_WR_DATA(fc);
				else LCD_WR_DATA(bc);
				m++;
				if(m % sizex == 0)
				{
					m = 0;
					break;
				}
			}
			else
			{
				if(temp & (0x01 << t))LCD_DrawPoint(x, y, fc);
				x++;
				if((x - x0) == sizex)
				{
					x = x0;
					y++;
					break;
				}
			}
		}
	}
}

void LCD_ShowString(u16 x, u16 y, const u8 *p, u16 fc, u16 bc, u8 sizey, u8 mode)
{
	while(*p != '\0')
	{
		LCD_ShowChar(x, y, *p, fc, bc, sizey, mode);
		x += sizey / 2;
		p++;
	}
}


u32 mypow(u8 m, u8 n)
{
	u32 result = 1;
	while(n--)result *= m;
	return result;
}


void LCD_ShowIntNum(u16 x, u16 y, u16 num, u8 len, u16 fc, u16 bc, u8 sizey)
{
	u8 t, temp;
	u8 enshow = 0;
	u8 sizex = sizey / 2;
	for(t = 0; t < len; t++)
	{
		temp = (num / mypow(10, len - t - 1)) % 10;
		if(enshow == 0 && t < (len - 1))
		{
			if(temp == 0)
			{
				LCD_ShowChar(x + t * sizex, y, ' ', fc, bc, sizey, 0);
				continue;
			}
			else enshow = 1;

		}
		LCD_ShowChar(x + t * sizex, y, temp + 48, fc, bc, sizey, 0);
	}
}


void LCD_ShowFloatNum1(u16 x, u16 y, float num, u8 len, u16 fc, u16 bc, u8 sizey)
{
	u8 t, temp, sizex;
	u16 num1;
	sizex = sizey / 2;
	num1 = num * 100;
	for(t = 0; t < len; t++)
	{
		temp = (num1 / mypow(10, len - t - 1)) % 10;
		if(t == (len - 2))
		{
			LCD_ShowChar(x + (len - 2)*sizex, y, '.', fc, bc, sizey, 0);
			t++;
			len += 1;
		}
		LCD_ShowChar(x + t * sizex, y, temp + 48, fc, bc, sizey, 0);
	}
}

void LCD_Initial(void)
{ 
	u16 y_len =400;
	u16 y_len1 =200;
	u16 y_len300 =300;
	u16 y_len100 =100;
	u16 x_len=24;
	u16 x_1=12;
	u16 x_0=0;
	u16 x_6=6;
	u16 x_18=18;
	LCD_Fill(0, 0, 128, 160, BLACK);
	LCD_DrawRectangle(18, 53, 122, 148, LGRAY);
	LCD_ShowIntNum(0,117,y_len100,3,LGRAY,BLACK,12);
	LCD_ShowIntNum(0,67,y_len300,3,LGRAY,BLACK,12);
	LCD_ShowIntNum(0,92,y_len1,3,LGRAY,BLACK,12);
	LCD_ShowIntNum(12,150,x_0,1,LGRAY,BLACK,12);
	LCD_ShowIntNum(42,150,x_6,1,LGRAY,BLACK,12);
	LCD_ShowIntNum(68,150,x_1,2,LGRAY,BLACK,12);
	LCD_ShowIntNum(92,150,x_18,2,LGRAY,BLACK,12);
	LCD_ShowIntNum(117,150,x_len,2,LGRAY,BLACK,12);
	LCD_DrawLine(80,63,95,63,RED);
	LCD_ShowString(50,57,"P",BLUE,BLACK,12,0);
	LCD_DrawLine(30,63,45,63,BLUE);
	LCD_ShowString(100,57,"T",RED,BLACK,12,0);
}


void LCD_round(int i)
{
		char num[10];
		LCD_Fill(0, 0, 128, 12, BLACK);
		LCD_ShowString(0,0,"Round ",YELLOW,BLACK,12,0);
		sprintf(num, "%d", i);
		LCD_ShowString(36,0,num ,YELLOW,BLACK,12,0);
		LCD_ShowString(54,0," Training" ,YELLOW,BLACK,12,0);
		
}


void LCD_batch(int i)
{
		char num[10];
		LCD_Fill(0, 12, 128, 24, BLACK);
		LCD_ShowString(0,12,"Batch ",YELLOW,BLACK,12,0);
		sprintf(num, "%d", i);
		LCD_ShowString(36,12,num ,YELLOW,BLACK,12,0);
		LCD_ShowString(54,12," Completed" ,YELLOW,BLACK,12,0);
		LCD_ShowString(0,24,"Time Usage:",YELLOW,BLACK,12,0);
}

void LCD_time_record(int i, int timecost)
{
		char num[10];
		char numb[20];
		int cost_time=timecost/10 + costtime*1000;
		LCD_Fill(0, 24, 128, 36, BLACK);
		LCD_ShowString(0,24,"Epoch ",WHITE,BLACK,12,0);
		sprintf(num, "%d", i);
		LCD_ShowString(36,24,num ,WHITE,BLACK,12,0);
		LCD_ShowString(54,24,":" ,WHITE,BLACK,12,0);
		sprintf(numb, "%d", cost_time);
		LCD_ShowString(66,24,numb ,WHITE,BLACK,12,0);
		LCD_ShowString(102,24,"ms" ,WHITE,BLACK,12,0);
		costtime =0;
}


void LCD_result(void)
{
		
		int Show_truei[Output_dimension] ={0};	
		int Show_predi[Output_dimension] ={0};
		LCD_Fill(0, 0, 126, 48, BLACK);
		LCD_Fill(19, 69, 122, 147,BLACK);
		LCD_DrawLine(19,73,122,73,LGRAY);
		LCD_DrawLine(19,98,122,98,LGRAY);
		LCD_DrawLine(19,123,122,123,LGRAY);
		LCD_DrawLine(45,73,45,148,LGRAY);
		LCD_DrawLine(71,73,71,148,LGRAY);
		LCD_DrawLine(97,73,97,148,LGRAY);
		LCD_ShowString(0,0,"Pred:",YELLOW,BLACK,12,0);
		LCD_ShowString(0,24,"True:",YELLOW,BLACK,12,0);
		
		for(int j = 0; j<Output_dimension;j++)
		{
			 Show_predi[j] = (int)(Prediction_show[j]/4); 
       Show_truei[j] = (int)(True_show[j]/4);  			    
		}
		for(int j = 0; j<Output_dimension-1;j++)
		{			
				if(Show_predi[j]>1){
					LCD_DrawLine(24+4*j, 148-Show_truei[j],28+4*j, 148-Show_truei[j+1], BLUE);
					LCD_DrawLine(24+4*j, 148-Show_predi[j],28+4*j, 148-Show_predi[j+1], RED);}
		}
		for(int i=0; i<Output_dimension;i++)
		{		
				if(i<2){
						if(Prediction_inverse[i]<100)
						{
								LCD_ShowFloatNum1(40+40*i, 0, Prediction_inverse[i], 4, YELLOW, BLACK, 12);
						}else{
								LCD_ShowFloatNum1(40+40*i, 0, Prediction_inverse[i], 5, YELLOW, BLACK, 12);
						}
						
						if(Label_inverse[i]<100)
					 {
							LCD_ShowFloatNum1(40+40*i, 24, Label_inverse[i], 4, YELLOW, BLACK, 12);
						}else{
							LCD_ShowFloatNum1(40+40*i, 24, Label_inverse[i], 5,YELLOW, BLACK, 12);
					 }
				}else{
					if(Prediction_inverse[i]<100)
					{
							LCD_ShowFloatNum1(40*i-40, 12, Prediction_inverse[i], 4, YELLOW, BLACK, 12);
					}else{
							LCD_ShowFloatNum1(40*i-40, 12, Prediction_inverse[i], 5, YELLOW, BLACK, 12);
					}
					
					if(Label_inverse[i]<100)
					 {
							LCD_ShowFloatNum1(40*i-40, 36, Label_inverse[i], 4, YELLOW, BLACK, 12);
						}else{
							LCD_ShowFloatNum1(40*i-40, 36, Label_inverse[i], 5, YELLOW, BLACK, 12);
					 }
				
				}
				
				
		}
					
}

			
void LCD_indic(void)
{
		LCD_Fill(0, 0, 128, 48, BLACK);
		LCD_ShowString(0,0,"MSE:",WHITE,BLACK,12,0);
		LCD_ShowString(0,12,"RMSE:",WHITE,BLACK,12,0);
		LCD_ShowString(0,24,"MAE:",WHITE,BLACK,12,0);
		LCD_ShowString(0,36,"MAPE:",WHITE,BLACK,12,0);
		LCD_ShowFloatNum1(30, 0, mse, 6, WHITE, BLACK, 12);
		LCD_ShowFloatNum1(30, 12, rmse, 6, WHITE, BLACK, 12);
		LCD_ShowFloatNum1(30, 24, mae, 6, WHITE, BLACK, 12);
		LCD_ShowFloatNum1(30, 36, mape, 6, WHITE, BLACK, 12);
		
}



		