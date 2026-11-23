
import matplotlib.pyplot as plt

mic0 = []

for line in open('data/data_0_0.txt'):
    mic0.append(int(line))

mic1 = []

for line in open('data/data_0_1.txt'):
    mic1.append(int(line))

t = []

for i in range(len(mic1)):
    t.append(i)

def make_plot(x, y1, y2):
    plt.figure(figsize=(10,6))
    plt.plot(x, y1, "red")
    plt.plot(x, y2, "blue")
    plt.show()

make_plot(t, mic0, mic1)