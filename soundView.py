"""Просмотр записей микрофонов: начало импульса и ручная отметка задержки.

Запуск из корня репозитория:
    python3 sound/scripts/soundView.py
Название записи задаётся переменной name (air1, breath1, ...).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

dataPath = "data/"
name = "AIR7hlop"   # какую запись смотрим: air1, breath1, ...

mic0 = np.loadtxt(dataPath + name + "_0.txt")
mic1 = np.loadtxt(dataPath + name + "_1.txt")

mic0 = mic0 - np.mean(mic0[:200])   # вычитаем фон, первые 200 отсчётов до импульса
mic1 = mic1 - np.mean(mic1[:200])

time0 = np.arange(len(mic0)) * 0.004   # время отсчётов, мс: шаг 4 мкс
time1 = time0 + 0.002                  # микрофон 1 измеряется на 2 мкс позже

plt.figure(figsize=(10, 6))
plt.plot(time0, mic0, label="микрофон 0")
plt.plot(time1, mic1, label="микрофон 1")
plt.xlabel("Время, мс")
plt.ylabel("Сигнал (коды АЦП за вычетом фона)")
plt.title("Запись %s" % name)
plt.legend()
plt.grid()
plt.show()
