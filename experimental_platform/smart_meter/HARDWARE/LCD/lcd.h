#ifndef __LCD_H
#define __LCD_H
#include "sys.h"


void LCD_Fill(u16 xsta, u16 ysta, u16 xend, u16 yend, u16 color); //指定区域填充颜色
void LCD_DrawPoint(u16 x, u16 y, u16 color); //在指定位置画一个点
void LCD_DrawLine(u16 x1, u16 y1, u16 x2, u16 y2, u16 color); //在指定位置画一条线
void LCD_DrawRectangle(u16 x1, u16 y1, u16 x2, u16 y2, u16 color); //在指定位置画一个矩形
void Draw_Circle(u16 x0, u16 y0, u8 r, u16 color); //在指定位置画一个圆

void LCD_ShowChar(u16 x, u16 y, u8 num, u16 fc, u16 bc, u8 sizey, u8 mode); //显示一个字符
void LCD_ShowString(u16 x, u16 y, const u8 *p, u16 fc, u16 bc, u8 sizey, u8 mode); //显示字符串
u32 mypow(u8 m, u8 n); //求幂
void LCD_ShowIntNum(u16 x, u16 y, u16 num, u8 len, u16 fc, u16 bc, u8 sizey); //显示整数变量
void LCD_ShowFloatNum1(u16 x, u16 y, float num, u8 len, u16 fc, u16 bc, u8 sizey); //显示两位小数变量
//字库函数
void ZK_command(u8 dat);
u8 get_data_from_ROM(void);
void get_n_bytes_data_from_ROM(u8 AddrHigh, u8 AddrMid, u8 AddrLow, u8 *pBuff, u8 DataLen);
void Display_GB2312(u16 x, u16 y, u8 zk_num, u16 fc, u16 bc);
void Display_GB2312_String(u16 x, u16 y, u8 zk_num, u8 text[], u16 fc, u16 bc);
void Display_Asc(u16 x, u16 y, u8 zk_num, u16 fc, u16 bc);
void Display_Asc_String(u16 x, u16 y, u16 zk_num, u8 text[], u16 fc, u16 bc);
void Display_Arial_TimesNewRoman(u16 x, u16 y, u8 zk_num, u16 fc, u16 bc);
void Display_Arial_String(u16 x, u16 y, u16 zk_num, u8 text[], u16 fc, u16 bc);
void Display_TimesNewRoman_String(u16 x, u16 y, u16 zk_num, u8 text[], u16 fc, u16 bc);

void LCD_round(int i);
void LCD_batch(int i);
void LCD_time_record(int i, int timecost);
void LCD_result(void);
void LCD_Initial(void);
//void LCD_indic(void);

//画笔颜色
#define WHITE         	 0xFFFF
#define BLACK         	 0x0000
#define BLUE           	 0x001F
#define BRED             0XF81F
#define GRED 			       0XFFE0
#define GBLUE			       0X07FF
#define RED           	 0xF800
#define MAGENTA       	 0xF81F
#define GREEN         	 0x07E0
#define CYAN          	 0x7FFF
#define YELLOW        	 0xFFE0
#define BROWN 			     0XBC40 //棕色
#define BRRED 			     0XFC07 //棕红色
#define GRAY  			     0X8430 //灰色
#define DARKBLUE      	 0X01CF	//深蓝色
#define LIGHTBLUE      	 0X7D7C	//浅蓝色  
#define GRAYBLUE       	 0X5458 //灰蓝色
#define LIGHTGREEN     	 0X841F //浅绿色
#define LGRAY 			     0XC618 //浅灰色(PANNEL),窗体背景色
#define LGRAYBLUE        0XA651 //浅灰蓝色(中间层颜色)
#define LBBLUE           0X2B12 //浅棕蓝色(选择条目的反色)
 

#endif





