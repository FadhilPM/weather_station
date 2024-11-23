#!/usr/bin/python -u
import RPi.GPIO as GPIO
import serial
import time, struct
import schedule
import Adafruit_DHT
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import os;

#############################################
"INPUT SECTION"
#############################################

delay = 10                                  #interval of recording data (in seconds) used in the function 'recording()' below

"""Change filename"""

filename = "DHT22_CALIBRATION_2"                      #<< Change the filename

"""Change filename"""

#############################################
"CHANNEL CREATION OF HUMIDITY AND TEMPERATURE DHT22 SENSOR"
#############################################

DHT_SENSOR = Adafruit_DHT.DHT22
DHT_PIN_FIRST = 4                               ### Pin number can change with your preference, check GPIO pinout for reference
DHT_PIN_SECOND = 10
#############################################
"FUNCTIONS FOR READING AND RECORDING SENSORS"
#############################################

def weather():
    try:
##       {Choice of interval selection to run script}
        """Best to keep interval in whole 'hour' format as seen below."""
        """Graph plotting script handles data in hours exclusively"""
        """Change the schedule only if you are mindful of data processing"""
        """e.g at the 15/30/45th minute of every hour works too."""
        """Any other denomination would be tedious though i.e last 3 options"""

##        schedule.every().hour.at(":30").do(recording)
#         schedule.every().hour.at(":00").do(recording)
        
        print("recording data...")
#         schedule.every(1).minutes.do(recording)
##        schedule.every().day.do(recording)
        schedule.every(10).seconds.do(recording)
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt: # If CTRL+C is pressed, exit cleanly:
        print("Program end")

def recording():
    humidity,temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN_FIRST)
    humidity_2, temperature_2 = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN_SECOND);
    
    """BASED ON YOUR CALIBRATION OR THE DETERMINATION OF ERROR, CHANGE THE HUMIDITY AND TEMPERATURE ACCORDINGLY BY ADDING AN EQUATION"""
    """e.g HUM = humidity + 0.3. Then change the defined humidity and temperature terms in the script below respectively"""
    
    if humidity is not None and temperature is not None:
        f = open(filename+".csv", "a")
        m = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        x = datetime.strptime(m, '%Y-%m-%d %H:%M:%S')
        xs = matplotlib.dates.date2num(x)
        h = datetime.now().hour

        
        """ Change humidity and temperature according to conclusions from calibration"""
        
        data_pt = "{:1}{:15.6f}{:7.2f}{:7.2f}{:7.2f}{:7.2f}\n".format(h, xs, temperature, temperature_2, humidity, humidity_2)
        
        """ Change humidity and temperature according to conclusions from calibration"""
        
        ###############################################
        ###Format of {} above:###
        ###{'Column/printing index':'Space between columns'.'Number of decimal places'f} , f = float i.e numbers with decimals ###
        ###############################################
        f.write(data_pt)
        f.flush(); #forcibly flush to memory
        os.fsync(f.fileno()); #forcibly flush from memory to disk 
        f.close(); #paranoia
        print(m)
        print("Hour= {:1} Temp_1= {:1.2f}*C  Temp_2= {:1.2f}*C  Humidity_1= {:1.2f}*C  Humidity_2= {:1.2f}%".format(h,temperature,temperature_2,humidity,humidity_2))
        print("delta_T= {:1.4f}*C delta_h= {:1.4f}%".format(temperature - temperature_2, humidity-humidity_2));
        #time.sleep(delay) #why, exactly, are we doing this?
    else:
        print("Problem occurred with DHT22 sensor")

#############################################
"EXECUTION OF WEATHER STATION"
#############################################
weather()

