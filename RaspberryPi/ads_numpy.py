import spidev
import time
import RPi.GPIO as gpio
import os
import csv
import numpy as np
import struct
from numpy import asarray
from numpy import savetxt
#global val
#val=[]


class AQ():
    

    def __init__(self, spi_bus=0, spi_device=0):
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.mode = 0b00
        self.spi.max_speed_hz = 10000000
        self.spi.bits_per_word = 8
        self.spi.threewire = False
        self.spi.cshigh = True
        self.spi.lsbfirst = False
        #PIN=23        
        #PIN2=27
        
        

        
    def get_register(self):           
        val=[i*0 for i in range(1,28673)]
        print (val)
        val_ret = self.spi.xfer3(val)
        print (val_ret)
        return val_ret
                   
    def archive(self, name):               
        if not os.path.exists(name):
            return name
        base, extension = os.path.splitext(name)
        number = 1
        while True:
            name = '{}_{:03d}{}'.format(base, number, extension)
            if not os.path.exists(name):
                return name
            number += 1
            
    def dim_matrix(self, mtx):
        d=np.array([])
        d=mtx.shape
        return d
            
    def int_to_bol(self, h, l):
        W=(h*256)+l
        if W>32767:
            W = W-65536              
        valor= (W/32768)*10
        return valor
                    
#class interrupt():
    
 #   def __init__(self):
 #       PIN=23        
 #       PIN2=17
        
        
 #       gpio.setmode(gpio.BCM)
 #       #gpio.setup(PIN2, gpio.OUT)
 #       gpio.setup(PIN, gpio.IN, pull_up_down = gpio.PUD_DOWN)
 #       gpio.add_event_detect(PIN, gpio.RISING, callback = get_register)
    
   # def aqst(self):
   #     gpio.output(27, gpio.HIGH)
   #     gpio.output(27, gpio.LOW)
        
