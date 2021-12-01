import spidev
import ads_numpy
import numpy as np
import RPi.GPIO as gpio
import tools
import pickle as pkl

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

binary_data = []

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

    binary_data.append(spi_output)

    print("time serie %d" % (i+1))

    print ("spi_output")

    print(np.info(spi_output))

    print(20*'=')

    # raspberry sends to STM32 the acquisition signal

    gpio.output(acquisition_command, gpio.HIGH)

# formatting binary array

binary_data = np.array(binary_data)

binary_data = np.reshape (binary_data, (-1, 16))

# applying tranference_function

data = tools.transference_function(binary_data)

# saving results

filename = (
    'acquisition_files/' + 
    tools.verification_tool.time_stamp() +
    '__time_serie.pkl'
)

fileObject = open(filename, 'wb')

pkl.dump(data, fileObject)

fileObject.close()


gpio.cleanup()