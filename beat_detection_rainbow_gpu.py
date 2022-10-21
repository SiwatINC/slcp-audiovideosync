import pyaudio
import numpy as np
from time import sleep
import cupy
import atexit
import signal
from siwat_light_control_protocol.siwat_light_control_protocol_multi_serial import siwat_light_control_protocol_multi_serial as slcp
import colorsys

def sigint_handler(signal=None, frame=None):
    print ('KeyboardInterrupt is caught')
    leds.turn_off()
    sleep(0.75)
    exit()
signal.signal(signal.SIGINT, sigint_handler)
atexit.register(sigint_handler)

SERIAL_PORTS = ["COM3"] # Change if neccessary
LED_MAP = [60] # Change if neccessary, number of LEDs
leds = slcp(SERIAL_PORTS,LED_MAP)
leds.turn_off()

SAMPLE_SIZE = 4096
SAMPLE_RATE = 48000
LOWPASS_CUTOFF = 50
AMPLITUDE_MULTIPLIER = 0.1
VELOCITY = 6

audio = pyaudio.PyAudio()

audioStream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=SAMPLE_SIZE)

pastBassSignal = [0]*100
timecounter = 0
while True:
    data = audioStream.read(SAMPLE_SIZE)
    sample = cupy.frombuffer(data, dtype=np.int16)
    power = cupy.sum(cupy.abs(sample))/SAMPLE_SIZE
    if power > 1000:
        freq_dom = cupy.fft.rfft(sample,10000)
        freqs = cupy.fft.rfftfreq(len(freq_dom))
        power_bass = cupy.sum(cupy.abs(freq_dom[0:LOWPASS_CUTOFF]))/cupy.sum(cupy.abs(freq_dom))*power
        power_bass = max(0,power_bass-250)
        if len(pastBassSignal)>100:
            pastBassSignal.pop(0)
        pastBassSignal.append(power_bass)
        print(power_bass)

        idmax = cupy.argmax(cupy.abs(freq_dom))
        freqmax = abs(freqs[idmax]*SAMPLE_RATE)
        brightness = int(min(255,power_bass*AMPLITUDE_MULTIPLIER))
        for j in range(0,sum(LED_MAP)):
            r, g, b = colorsys.hsv_to_rgb(((-timecounter*VELOCITY+j*4)%360)/360,1,1)
            leds.set_led_at(j,r=int(r*brightness),g=int(g*brightness),b=int(b*brightness))
        timecounter+=1
        leds.show()
    else:
        leds.turn_off()
audioStream.stop_stream()
audioStream.close()
audio.terminate()