import spidev
import ads_numpy
import numpy as np
import RPi.GPIO as gpio
import tools
import pickle as pkl

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
spi.max_speed_hz = 10000000
spi.bits_per_word = 8
spi.threewire = False
spi.cshigh = True
spi.lsbfirst = False


### Acquisition routine

# raspberry sends to STM32 the acquisition signal

gpio.output(acquisition_command, gpio.HIGH)

# creating a dictionary for receiving the data

data_dict = {}

# Initializing progress bar

bar = tools.ProgBar(num_time_series,'\nAcquisition Process')

for i in range(num_time_series):

    # waiting STM32 signal to start SPI communication

    while True:
        if gpio.event_detected(spi_init_flag):
            break

    bar.message("start SPI communication received")
    
    # raspberry sends to STM32 the busy signal 

    gpio.output(acquisition_command, gpio.LOW)

    # creating empty array for establishing SPI communication

    dummy_array = [i for i in range(num_bytes)]

    spi_output = spi.xfer3(dummy_array)

    spi_output = np.array(spi_output,dtype=np.uint8)

    data_dict[i] = spi_output
    
    bar.message("data acquisition complete")
    
    # raspberry sends to STM32 the acquisition signal

    gpio.output(acquisition_command, gpio.HIGH)

    bar.update()

# Initializing progress bar

bar = tools.ProgBar(len(data_dict),'\nApplying tranference function')

for key in data_dict:

    # formatting binary array

    binary_data = np.reshape (data_dict[key], (-1, 16))

    # applying tranference_function

    data = tools.transference_function(binary_data)

    data_dict[key] = data

    bar.update()


# Initializing progress bar

bar = tools.ProgBar(len(data_dict),'\nSaving Time Series')

for key in data_dict:

    # saving results

    filename = (
        'acquisition_files/' + 
        tools.time_stamp() +
        '__time_serie.pkl'
    )

    with open(filename, "wb") as file_object:

        pkl.dump(data_dict[key], file_object)

    bar.update

gpio.cleanup()