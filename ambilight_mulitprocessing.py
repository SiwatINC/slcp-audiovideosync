import multiprocessing
import pyautogui
import scipy.cluster as cluster
import numpy as np
from time import sleep
import numpy
import signal
import atexit
from time import perf_counter as timestamp
from siwat_light_control_protocol.siwat_light_control_protocol_multi_serial import siwat_light_control_protocol_multi_serial as slcp
from multiprocessing import Pool
import PIL.ImageGrab
import colorsys
from scipy.stats import mode

BOARDER_SIZE = 300
TOP_LEDS = 19
RIGHT_LEDS = 11
BUTTOM_LEDS = 19
LEFT_LEDS = 11
NUM_LEDS = TOP_LEDS+RIGHT_LEDS+BUTTOM_LEDS+LEFT_LEDS
MIN_TIME = 0.05
METHOD = 'MEDIAN' #MEAN/MEDIAN/MODE/CLUSTER_MEAN, WARNING: CLUSTER_MEAN might burn down your house.
#CLUSTER_MEAN regonize multiple color in the sample and choose one that is the most dominant (try googling K-mean)
SATURATION_BOOST_FACTOR = 2
NUM_CLUSTERS = 3
KMEAN_QUALITY = 50 
NUM_THREADS = 8

if __name__ == '__main__':
    global leds
    SERIAL_PORTS = ["COM3"] # Change if neccessary
    LED_MAP = [NUM_LEDS] # Change if neccessary, number of LEDs
    leds = slcp(SERIAL_PORTS,LED_MAP)
    leds.turn_off()

def sigint_handler(signal=None, frame=None):
    print ('KeyboardInterrupt is caught')
    leds.turn_off()
    sleep(0.75)
    exit()

class size:
    width = None
    height = None

def get_screenshot():
    return PIL.ImageGrab.grab()


def find_dorminant_color(im):

    color = numpy.reshape(numpy.asarray(im),(-1,3))
    
    if METHOD == 'MODE':
        color = mode(color,axis=0).mode[0]
    elif METHOD == 'MEDIAN':
        color = np.median(color,axis=0)
    elif METHOD == 'MEAN':
        color = np.mean(color,axis=0)
    elif METHOD == 'CLUSTER_MEAN':
        im = im.resize((KMEAN_QUALITY, KMEAN_QUALITY))
        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
        codes, dist = cluster.vq.kmeans(ar, NUM_CLUSTERS)

        vecs, dist = cluster.vq.vq(ar, codes) 
        counts, bins = np.histogram(vecs, len(codes))

        index_max = np.argmax(counts)
        color = codes[index_max]

    h,s,v = colorsys.rgb_to_hsv(r=color[0]/255,g=color[1]/255,b=color[2]/255)
    s = min(1,s*SATURATION_BOOST_FACTOR)
    r,g,b = colorsys.hsv_to_rgb(h=h,s=s,v=v)
    return [r*255,g*255,b*255]
if __name__ == '__main__':
    size = size()
    size.width=pyautogui.size().width
    size.height=pyautogui.size().height
    signal.signal(signal.SIGINT, sigint_handler)
    atexit.register(sigint_handler)

    NUM_LEDS = TOP_LEDS+RIGHT_LEDS+BUTTOM_LEDS+LEFT_LEDS

    leds.turn_off()

    screenshot = pyautogui.screenshot()
    pool = Pool(NUM_THREADS)

    while True:
        try:
            screenshot = get_screenshot()
            lastTime = timestamp()
            chunk = []
            top = screenshot.crop(box=[0,0,size.width,BOARDER_SIZE])
            left = screenshot.crop(box=[0,0,BOARDER_SIZE,size.height])
            buttom = screenshot.crop(box=[0,size.height-BOARDER_SIZE,size.width,size.height])
            right = screenshot.crop(box=[size.width-BOARDER_SIZE,0,size.width,size.height])
            for i in range(0,TOP_LEDS):
                segment = top.crop(box=[i*size.width/TOP_LEDS,0,(i+1)*size.width/TOP_LEDS,BOARDER_SIZE])
                chunk.append(segment)

            for i in range(0,RIGHT_LEDS):
                segment = right.crop(box=[0,i*size.height/RIGHT_LEDS,BOARDER_SIZE,(i+1)*size.height/RIGHT_LEDS])
                chunk.append(segment)
            
            for i in reversed(range(0,BUTTOM_LEDS)):
                segment = buttom.crop(box=[i*size.width/BUTTOM_LEDS,0,(i+1)*size.width/BUTTOM_LEDS,BOARDER_SIZE])
                chunk.append(segment)

            for i in reversed(range(0,LEFT_LEDS)):
                segment = left.crop(box=[0,i*size.height/LEFT_LEDS,BOARDER_SIZE,(i+1)*size.height/LEFT_LEDS])
                chunk.append(segment)
            colors = pool.map(find_dorminant_color,chunk)
            for i in range(0,len(colors)):
                leds.set_led_at(i,r=int(colors[i][0]),g=int(colors[i][1]),b=int(colors[i][2]))
            leds.show()
            while timestamp()-lastTime < MIN_TIME:
                sleep(0.001)
            print("loop time : "+str(timestamp()-lastTime))
        except Exception as e:
            print(e)
            print("Retrying")
