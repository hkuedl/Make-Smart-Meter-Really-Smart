#ifndef __DATA_H
#define __DATA_H

#include "sys.h"

#define Mean  95.54131623085146f
#define Var   51.5454596997836f

extern const float Trainx[14008][5];
extern const float Trainy[14008];
extern const float Testx[3516][5];
extern const float Testy[3516];
extern const char Identifier_extractor_weight;
extern const char Identifier_auxiliary_weight;
extern const char Identifier_regressor_weight;
extern const char Identifier_extractor_bias;
extern const char Identifier_auxiliary_bias;
extern const char Identifier_regressor_bias;
extern const char Identifier_extractor_activation;
extern const char Identifier_processor_activation_gradient;
extern const char Identifier_extractor_output;
#endif
