/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2020 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define ext_nss_Pin GPIO_PIN_0
#define ext_nss_GPIO_Port GPIOA
#define ext_nss_EXTI_IRQn EXTI0_IRQn
#define Frsdt_Pin GPIO_PIN_1
#define Frsdt_GPIO_Port GPIOA
#define Cvrst_Pin GPIO_PIN_2
#define Cvrst_GPIO_Port GPIOA
#define Busy_Pin GPIO_PIN_3
#define Busy_GPIO_Port GPIOA
#define Busy_EXTI_IRQn EXTI3_IRQn
#define db0_Pin GPIO_PIN_0
#define db0_GPIO_Port GPIOB
#define db1_Pin GPIO_PIN_1
#define db1_GPIO_Port GPIOB
#define db2_Pin GPIO_PIN_2
#define db2_GPIO_Port GPIOB
#define db10_Pin GPIO_PIN_10
#define db10_GPIO_Port GPIOB
#define db11_Pin GPIO_PIN_11
#define db11_GPIO_Port GPIOB
#define db12_Pin GPIO_PIN_12
#define db12_GPIO_Port GPIOB
#define db13_Pin GPIO_PIN_13
#define db13_GPIO_Port GPIOB
#define db14_Pin GPIO_PIN_14
#define db14_GPIO_Port GPIOB
#define db15_Pin GPIO_PIN_15
#define db15_GPIO_Port GPIOB
#define Cs_Pin GPIO_PIN_8
#define Cs_GPIO_Port GPIOA
#define Os0_Pin GPIO_PIN_9
#define Os0_GPIO_Port GPIOA
#define rst_Pin GPIO_PIN_10
#define rst_GPIO_Port GPIOA
#define ce_Pin GPIO_PIN_11
#define ce_GPIO_Port GPIOA
#define Os1_Pin GPIO_PIN_12
#define Os1_GPIO_Port GPIOA
#define Os2_Pin GPIO_PIN_15
#define Os2_GPIO_Port GPIOA
#define db3_Pin GPIO_PIN_3
#define db3_GPIO_Port GPIOB
#define db4_Pin GPIO_PIN_4
#define db4_GPIO_Port GPIOB
#define db5_Pin GPIO_PIN_5
#define db5_GPIO_Port GPIOB
#define db6_Pin GPIO_PIN_6
#define db6_GPIO_Port GPIOB
#define db7_Pin GPIO_PIN_7
#define db7_GPIO_Port GPIOB
#define db8_Pin GPIO_PIN_8
#define db8_GPIO_Port GPIOB
#define db9_Pin GPIO_PIN_9
#define db9_GPIO_Port GPIOB
/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
