import spidev
import time
import ads_numpy
import os
import csv
import numpy as np
from numpy import asarray
from numpy import savetxt
import RPi.GPIO as gpio

ad = ads_numpy.AQ()

#### User set variables

spi_init_flag = 23

acquisition_command =27

num_time_series = 2

num_bytes = 28672

# cleaning residual state of the GPIO

gpio.cleanup()

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
spi.max_speed_hz = 10000000
spi.bits_per_word = 8
spi.threewire = False
spi.cshigh = True
spi.lsbfirst = False

### Acquisition routine

# raspberry sends to STM32 the acquisition signal

gpio.output(acquisition_command, gpio.HIGH)

# creating list for receiving the data

data = []

for i in range(num_time_series):

    # waiting STM32 signal to start SPI communication

    while True:
        if gpio.event_detected(spi_init_flag):
            break
    
    # raspberry sends to STM32 the busy signal 

    gpio.output(acquisition_command, gpio.LOW)

    # creating empty array for establishing SPI communication

    dummy_array = [i for i in range(num_bytes)]

    spi_output = spi.xfer3(dummy_array)

    spi_output = np.array(spi_output,dtype=np.unint8)

    data.append(spi_output)

    print("time serie %d" % (i+1))

    print ("spi_output")

    print(np.info(spi_output))

    print(20*'=')

    # raspberry sends to STM32 the acquisition signal

    gpio.output(acquisition_command, gpio.HIGH)


data = np.array(data)

data = np.reshape (data, (-1, 16))

#savetxt(name, data, fmt='%u', delimiter=',')

dim=np.array([])
dim= data.shape
print (dim)
n=dim[0]
valor=np.empty([n,8], dtype = float)
a=0
for a in range (n):
    b=0
    for b in range (0,15,2):

        hi=data[a,b]

        lo=data[a,(b+1)]

        c =(b/2)

        c=round(c)

        v=ad.int_to_bol(hi,lo)

        valor[a,c]=v
    a=a+1

print(type(valor))


gpio.cleanup()