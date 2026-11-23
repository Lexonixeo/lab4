
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundPico as pico

name = sys.argv[1]          # air1, breath1, ...
dataPath = "data/"
threshold = 20 

# mic0 - ЛЕВЫЙ (БЛИЖАЙШИЙ К ХЛОПКУ)

# GO_pin = 17
# is_sampled_pin = 27
# ch_select_pin = 22
# is_written_pin = 23

def make_plot(x, y1, y2):
    plt.figure(figsize=(10,6))
    plt.plot(x, y1, "red")
    plt.plot(x, y2, "blue")
    plt.show()

def init():
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(GO_pin, GPIO.OUT, initial=0)
    # GPIO.setup(is_sampled_pin, GPIO.IN)
    # GPIO.setup(ch_select_pin, GPIO.OUT, initial=0)
    # GPIO.setup(is_written_pin, GPIO.IN)

    pico.initPico()

    freq = 4 * 10**(-6)

    try: 
        mic0, mic1 = pico.capture(threshold)
    finally:
        pico.deinitPico()

    np.savetxt(dataPath + name + "_0.txt", mic0, fmt="%d")
    np.savetxt(dataPath + name + "_1.txt", mic1, fmt="%d")

    time = [0] * len(mic0)

    for i in range(1, len(time)):
        time[i] = i

    make_plot(time, mic0, mic1) 
    
def deinit():
    pico.deinitPico()

def main():
    init()
    deinit()

if __name__ == "__main__":
    main()