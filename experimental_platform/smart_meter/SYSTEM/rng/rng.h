#ifndef __BASIC_RNG_H
#define	__BASIC_RNG_H

#include "stm32f4xx.h"

u8 RNG_Init(void);
u32 RNG_Get_RandomNum(void);
int RNG_Get_RandomRange(int min,int max);




#endif /* __BASIC_RNG_H */

