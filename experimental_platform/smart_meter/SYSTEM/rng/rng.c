#include "rng.h"
#include "delay.h"
#include "sys.h"

//返回0 初始化成功，返回1初始化失败
u8 RNG_Init(void)    
{
    u16 i;
	  delay_init(168);
    //使能RNG时钟
    RCC_AHB2PeriphClockCmd(RCC_AHB2Periph_RNG,ENABLE);  //使能RNG时钟，在AHB2总线上
    //使能RNG
    RNG_Cmd(ENABLE);//使能RNG
    while(RNG_GetFlagStatus(RNG_FLAG_DRDY)==0){  //等待DRDY稳定，稳定之后不为0，返回1     
        i++;
        delay_us(100);                
        if(i >= 10000){
            return 1;       //超时强制返回
        }         
    }
    return 0;     
}


//读取数值函数
u32 RNG_Get_RandomNum(void)
{
    while(RNG_GetFlagStatus(RNG_FLAG_DRDY)==0);   //等待稳定
    return RNG_GetRandomNumber();    //获取并返回数值
}


//生成[min,max]范围的随机数，让随机数除以区间长度取余数，再加上min得到随机数
int RNG_Get_RandomRange(int min,int max)
{
  return RNG_Get_RandomNum()%(max-min+1)+min;
}

