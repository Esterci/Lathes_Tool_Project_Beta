import spidev
import RPi.GPIO as gpio
import tools
import os

# cleaning residual state of the GPIO

gpio.cleanup()

#### User set variables

spi_init_flag = 23

acquisition_command =27

num_time_series = 2

num_bytes = 28672

# setting comunication GPIOS

gpio.setmode(gpio.BCM)
gpio.setup(acquisition_command, gpio.OUT)
gpio.output(acquisition_command, gpio.LOW)
gpio.setup(spi_init_flag, gpio.IN, pull_up_down = gpio.PUD_DOWN)
gpio.add_event_detect(spi_init_flag, gpio.RISING)

# setting SPI parameters

spi = spidev.SpiDev()
spi.open(0, 0)
spi.mode = 0b00
spi.max_speed_hz = int(250e6/32)
spi.bits_per_word = 8
spi.threewire = False
spi.cshigh = True
spi.lsbfirst = False


### Acquisition routine

# Initializing progress bar

bar = tools.ProgBar(num_time_series,'\nAcquisition Process')

for i in range(num_time_series):

    # raspberry sends to STM32 the acquisition signal

    gpio.output(acquisition_command, gpio.HIGH)

    # waiting STM32 signal to start SPI communication

    while True:
        if gpio.event_detected(spi_init_flag):
            break

    bar.message("start SPI communication received")
    
    # raspberry sends to STM32 the busy signal 

    gpio.output(acquisition_command, gpio.LOW)

    bar.update()

input("Insert SD card and press ENTER")

# removing previewous results

os.remove("figures/*") 
os.remove("resports/*") 

# initializing verification tool

tool = tools.verification_tool('acquisition_files/*')

# checking ranges

tool.range_check()

# writing reports

tool.write_report()

# ploting time series

tool.plot_time_series()

# ploting FFTs

tool.plot_fft()

