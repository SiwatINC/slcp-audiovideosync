import pyautogui
import scipy.cluster as cluster
import scipy
import sys
import numpy as np
from time import sleep, time
import cupy
import signal
import atexit
from time import perf_counter as timestamp
from siwat_light_control_protocol.siwat_light_control_protocol_multi_serial import siwat_light_control_protocol_multi_serial as slcp
import threading

global leds
def sigint_handler(signal=None, frame=None):
    print ('KeyboardInterrupt is caught')
    leds.turn_off()
    sleep(0.75)
    exit()

signal.signal(signal.SIGINT, sigint_handler)
atexit.register(sigint_handler)

FAST_MODE = True
global NUM_LEDS,BOARDER_SIZE,TOP_LEDS,RIGHT_LEDS,BUTTOM_LEDS,LEFT_LEDS,screenshot

BOARDER_SIZE = 300
TOP_LEDS = 19
RIGHT_LEDS = 11
BUTTOM_LEDS = 17
LEFT_LEDS = 10
NUM_LEDS = TOP_LEDS+RIGHT_LEDS+BUTTOM_LEDS+LEFT_LEDS

SERIAL_PORTS = ["COM3"] # Change if neccessary
LED_MAP = [NUM_LEDS]
leds = slcp(SERIAL_PORTS,LED_MAP)
leds.turn_off()

lastTime = timestamp()

def find_dorminant_color(im):
    if FAST_MODE:
        color = cupy.reshape(cupy.asarray(im),(-1,3))
        color = cupy.median(color,axis=0)
        
        return color
    else:
        NUM_CLUSTERS = 5

        im = im.resize((150, 150))
        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
        codes, dist = cluster.vq.kmeans(ar, NUM_CLUSTERS)

        vecs, dist = cluster.vq.vq(ar, codes) 
        counts, bins = np.histogram(vecs, len(codes))

        index_max = np.argmax(counts)
        peak = codes[index_max]

        return peak

def processTopLeds():
    global NUM_LEDS,BOARDER_SIZE,TOP_LEDS,RIGHT_LEDS,BUTTOM_LEDS,LEFT_LEDS,leds,screenshot
    
    while True:
        top = screenshot.crop(box=[0,0,size.width,BOARDER_SIZE])
        for i in range(0,TOP_LEDS):
            segment = top.crop(box=[i*size.width/TOP_LEDS,0,(i+1)*size.width/TOP_LEDS,BOARDER_SIZE])
            colors = [int(color) for color in find_dorminant_color(segment)]
            leds.set_led_at(i,r=colors[0],g=colors[1],b=colors[2])
        leds.show()
def processLeftLeds():
    global NUM_LEDS,BOARDER_SIZE,TOP_LEDS,RIGHT_LEDS,BUTTOM_LEDS,LEFT_LEDS,board,screenshot
    while True:
        left = screenshot.crop(box=[0,0,BOARDER_SIZE,size.height])
        for i in range(0,LEFT_LEDS):
            segment = left.crop(box=[0,i*size.height/LEFT_LEDS,BOARDER_SIZE,(i+1)*size.height/LEFT_LEDS])
            colors = [int(color) for color in find_dorminant_color(segment)]
            leds.set_led_at(TOP_LEDS+RIGHT_LEDS+BUTTOM_LEDS+LEFT_LEDS-i,r=colors[0],g=colors[1],b=colors[2])
        leds.show()
def processButtomLeds():
    global NUM_LEDS,BOARDER_SIZE,TOP_LEDS,RIGHT_LEDS,BUTTOM_LEDS,LEFT_LEDS,board,screenshot
    while True:
        buttom = screenshot.crop(box=[0,size.height-BOARDER_SIZE,size.width,size.height])
        for i in range(0,BUTTOM_LEDS):
            segment = buttom.crop(box=[i*size.width/BUTTOM_LEDS,0,(i+1)*size.width/BUTTOM_LEDS,BOARDER_SIZE])
            colors = [int(color) for color in find_dorminant_color(segment)]
            leds.set_led_at(TOP_LEDS+RIGHT_LEDS+BUTTOM_LEDS-i,r=colors[0],g=colors[1],b=colors[2])
        leds.show()
def processRightLeds():
    global NUM_LEDS,BOARDER_SIZE,TOP_LEDS,RIGHT_LEDS,BUTTOM_LEDS,LEFT_LEDS,board,screenshot
    while True:
        right = screenshot.crop(box=[size.width-BOARDER_SIZE,0,size.width,size.height])
        for i in range(0,RIGHT_LEDS):
            segment = right.crop(box=[0,i*size.height/RIGHT_LEDS,BOARDER_SIZE,(i+1)*size.height/RIGHT_LEDS])
            colors = [int(color) for color in find_dorminant_color(segment)]
            leds.set_led_at(i+TOP_LEDS,r=colors[0],g=colors[1],b=colors[2])
        leds.show()
screenshot = pyautogui.screenshot()
size = pyautogui.size()
threading.Thread(target=processTopLeds).start()
threading.Thread(target=processLeftLeds).start()
threading.Thread(target=processRightLeds).start()
threading.Thread(target=processButtomLeds).start()
while True:
    screenshot = pyautogui.screenshot()
    
    
