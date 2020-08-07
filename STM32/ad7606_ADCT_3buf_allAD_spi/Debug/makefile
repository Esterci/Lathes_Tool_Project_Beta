################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include Middlewares/Third_Party/FreeRTOS/Source/portable/MemMang/subdir.mk
-include Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM3/subdir.mk
-include Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS_V2/subdir.mk
-include Middlewares/Third_Party/FreeRTOS/Source/subdir.mk
-include Drivers/STM32F1xx_HAL_Driver/Src/subdir.mk
-include Core/Startup/subdir.mk
-include Core/Src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
ad7606_ADCT_3buf_allAD_spi.elf \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
ad7606_ADCT_3buf_allAD_spi.list \

OBJCOPY_BIN += \
ad7606_ADCT_3buf_allAD_spi.bin \


# All Target
all: ad7606_ADCT_3buf_allAD_spi.elf secondary-outputs

# Tool invocations
ad7606_ADCT_3buf_allAD_spi.elf: $(OBJS) $(USER_OBJS) C:\Users\MarcosVictor\STM32CubeIDE\workspace_1.3.0\ad7606_ADCT_3buf_allAD_spi\STM32F103C8TX_FLASH.ld
	arm-none-eabi-gcc -o "ad7606_ADCT_3buf_allAD_spi.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m3 -T"C:\Users\MarcosVictor\STM32CubeIDE\workspace_1.3.0\ad7606_ADCT_3buf_allAD_spi\STM32F103C8TX_FLASH.ld" --specs=nosys.specs -Wl,-Map="ad7606_ADCT_3buf_allAD_spi.map" -Wl,--gc-sections -static --specs=nano.specs -mfloat-abi=soft -mthumb -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

ad7606_ADCT_3buf_allAD_spi.list: $(EXECUTABLES)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "ad7606_ADCT_3buf_allAD_spi.list"
	@echo 'Finished building: $@'
	@echo ' '

ad7606_ADCT_3buf_allAD_spi.bin: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "ad7606_ADCT_3buf_allAD_spi.bin"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) *
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_BIN)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
