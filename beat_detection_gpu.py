import pyaudio
import numpy as np
from time import sleep, time
import cupy
import atexit
import signal
from siwat_light_control_protocol.siwat_light_control_protocol_multi_serial import siwat_light_control_protocol_multi_serial as slcp

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

audio = pyaudio.PyAudio()

audioStream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=SAMPLE_SIZE)

pastBassSignal = [0]*100
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
        leds.set_led_at(r=int(min(255,power_bass*AMPLITUDE_MULTIPLIER)),g=0,b=0,auto_show=True)
    else:
        leds.turn_off()
audioStream.stop_stream()
audioStream.close()
audio.terminate()