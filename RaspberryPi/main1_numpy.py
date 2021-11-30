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
gpio.cleanup()
PIN=23
PIN2=27
gpio.setmode(gpio.BCM)
gpio.setup(PIN2, gpio.OUT)
gpio.output(PIN2, gpio.LOW)

gpio.setup(PIN, gpio.IN, pull_up_down = gpio.PUD_DOWN)
gpio.add_event_detect(PIN, gpio.RISING)
name = ad.archive("Aquisicao.csv");
canal=np.zeros((1,28672), dtype=np.uint8)

i=0
#gpio.output(PIN2, gpio.HIGH)
while(i<2):     
    print (5)
    
    if gpio.event_detected(PIN):                
        if (i==0):
            valor = ad.get_register()
            canal = np.array(valor, dtype=np.uint8)
            i=i+1
            #gpio.output(PIN2, gpio.LOW)
        else:
            valor = ad.get_register()     
            canais = np.array(valor, dtype=np.uint8)
            canal = np.vstack((canal,canais))
            i=i+1
            #gpio.output(PIN2, gpio.LOW)
        
        print (6)
else:
    print (canal)
    canal = np.reshape (canal, (-1, 16))
    savetxt(name, canal, fmt='%u', delimiter=',')
    dim=np.array([])
    dim= ad.dim_matrix(canal)
    print (dim)
    n=dim[0]
    valor=np.empty([n,8], dtype = float)
    a=0
    for a in range (n):
        b=0
        for b in range (0,15,2):
            hi=canal[a,b]
            lo=canal[a,(b+1)]
            c =(b/2)
            c=round(c)
            v=ad.int_to_bol(hi,lo)
            valor[a,c]=v
        a=a+1
    name = ad.archive("Aquisicao.csv");
    savetxt(name, valor, fmt='%f', delimiter=',')
    canal=np.zeros((1,28672), dtype=np.uint8)
    gpio.cleanup()

   

    
