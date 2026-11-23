"""Обработка данных лабораторной работы «Скорость звука в газовой среде».

Пункты 4.5 и 4.6 методички:
  шаг 1  — условия опыта из data/conditions.txt;
  шаг 2  — две расчётные зависимости скорости звука от доли CO2 и две прямые;
  шаг 3  — задержки звука между микрофонами по записям (находятся программой);
  шаг 4  — скорости звука и доли CO2 в атмосферном воздухе и в выдохе;
  шаг 5  — итоговый график plots/speed-of-sound.png.

Запуск из корня репозитория команды:
    python3 sound/scripts/soundProcessing.py
"""
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import soundFunctions as s

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

dataPath = "data/"
plotsPath = "plots/"
if not os.path.exists(plotsPath):
    os.makedirs(plotsPath)

L = 1.158            # расстояние между микрофонами, м
SAMPLE_MS = 0.004    # шаг отсчётов, мс (250 кГц)
MIC1_SHIFT_MS = 0.002  # микрофон 1 измеряется на 2 мкс позже микрофона 0


# ======================= Шаг 1 (п4.5). Условия опыта =======================
def readConditions(path=dataPath + "conditions.txt"):
    """Относительная влажность (%), температура (°C) из conditions.txt."""
    text = open(path, encoding="utf-8").read()
    rh = float(re.search(r"humidity\s*=\s*([-\d.]+)", text).group(1))
    t = float(re.search(r"Temperature\s*=\s*([-\d.]+)", text).group(1))
    return t, rh


t, rh = readConditions()
print("Условия опыта: t = %.1f °C, относительная влажность %.1f %%" % (t, rh))

# ================ Шаг 2 (п4.5). Две зависимости и две прямые ================
co2, speedRoom = s.speedOfSound(t, s.h2oFraction(t, rh), 5)
co2, speedWet = s.speedOfSound(t, s.h2oFraction(t, 100), 5)

# x — скорость звука, y — доля CO2: прямая сразу переводит скорость в долю
fitRoom = np.polyfit(speedRoom, co2, 1)
fitWet = np.polyfit(speedWet, co2, 1)

# Для графика кривые считаем до 7 %, чтобы на них попала и точка выдоха
# (её доля CO2 может выйти за 5 %). Прямые при этом остаются по 0–5 %, как в п4.5.
co2Plot, speedRoomPlot = s.speedOfSound(t, s.h2oFraction(t, rh), 7)
co2Plot, speedWetPlot = s.speedOfSound(t, s.h2oFraction(t, 100), 7)

print("\nПрямая, влажность аудитории: [%.3f  %.2f]" % (fitRoom[0], fitRoom[1]))
print("Прямая, влажность 100 %%:    [%.3f  %.2f]" % (fitWet[0], fitWet[1]))
print("  скорость при 0 %% CO2: %.2f м/с (влажность %.1f %%)" % (speedRoom[0], rh))
print("  скорость при 0 %% CO2: %.2f м/с (влажность 100 %%)" % speedWet[0])
print("  наклон: %.2f м/с на 1 %% CO2" % (speedRoom[0] - speedRoom[10]))

# ============ Шаг 3 (п4.6). Задержки по записям (программой) ============
def load(name):
    mic0 = np.loadtxt(dataPath + name + "_0.txt").astype(float)
    mic1 = np.loadtxt(dataPath + name + "_1.txt").astype(float)
    return mic0, mic1


def prepare(x):
    """Вычитает постоянный фон по первым 200 отсчётам до прихода импульса."""
    return x - np.mean(x[:200])


def smooth(x, w):
    """Скользящее среднее: давит шум АЦП, форму импульса сохраняет."""
    if w <= 1:
        return x.copy()
    return np.convolve(x, np.ones(w) / w, mode="same")


def detrend(y):
    """Убирает линейный тренд на участке: у каналов разный наклон фона."""
    n = len(y)
    tt = np.arange(n, dtype=float)
    A = np.vstack([tt, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def firstOnset(sig, raw):
    """Номер отсчёта, где сглаженный сигнал впервые уходит от фона на 6 шумов."""
    noise = np.std(raw[:400])
    thr = max(6 * noise, 12)
    return int(np.where(np.abs(sig) > thr)[0][0])


def findDelay(mic0, mic1, pre=-200, win=380, dsmooth=11, search=(600, 1100)):
    """Задержка по совмещению начала импульса двух микрофонов.

    Порядок из т4.4: вычитаем фон, сглаживаем, ищем сдвиг N отсчётов,
    при котором участок начала импульса совпадает лучше всего (максимум
    коэффициента корреляции). Возвращает N с долями отсчёта, tau и качество.
    """
    a = prepare(mic0)
    b = prepare(mic1)
    A = smooth(a, dsmooth)
    B = smooth(b, dsmooth)

    s0 = max(0, firstOnset(A, a) + pre)
    prof = []
    for N in range(*search):
        if s0 + N < 0:
            continue
        x = A[s0:s0 + win]
        y = B[s0 + N:s0 + N + win]
        if len(y) < win:
            break
        x = detrend(x)
        y = detrend(y)
        xc = x - x.mean()
        yc = y - y.mean()
        prof.append((N, float(xc @ yc / np.sqrt((xc @ xc) * (yc @ yc)))))
    prof = np.array(prof)

    k = int(np.argmax(prof[:, 1]))
    Nb = int(prof[k, 0])
    # уточнение вершины параболой по трём точкам — доли отсчёта
    if 0 < k < len(prof) - 1:
        y0, y1, y2 = prof[k - 1, 1], prof[k, 1], prof[k + 1, 1]
        den = y0 - 2 * y1 + y2
        dN = 0.5 * (y0 - y2) / den if den else 0.0
    else:
        dN = 0.0

    N = Nb + dN
    return dict(N=N, tau=N * SAMPLE_MS + MIC1_SHIFT_MS,
                cc=prof[k, 1], i0=firstOnset(A, a), a=a, b=b)


# Сетка параметров: задержку берём как медиану по сетке, а разброс по сетке —
# как оценку устойчивости алгоритма (аналог «навести мышь в разные точки»).
GRID = [(pre, win, ds) for pre in (-240, -200, -160)
        for win in (340, 380, 420) for ds in (9, 11, 13)]


def measureDelay(mic0, mic1):
    res = [findDelay(mic0, mic1, pre, win, ds) for pre, win, ds in GRID]
    N = np.array([r["N"] for r in res])
    best = res[int(np.argmax([r["cc"] for r in res]))]
    return dict(N=float(np.median(N)), tau=float(np.median(N)) * SAMPLE_MS + MIC1_SHIFT_MS,
                spread=float(N.std(ddof=1)) * SAMPLE_MS, cc=best["cc"],
                i0=best["i0"], a=best["a"], b=best["b"])


airFiles = ["AIR7hlop", "AIR8hlop", "AIR9hlop", "AIR10hlop", "AIR11hlop", "AIR12hlopclosed"]
breathFiles = ["breath1", "breath2", "breath3"]

print("\nЗадержки (найдены программой, без ручной отметки):")
print("  %-16s %8s %9s %8s %9s" % ("запись", "N", "tau, мс", "cc", "разброс"))
delays = {}
for group, files in (("air", airFiles), ("breath", breathFiles)):
    for name in files:
        mic0, mic1 = load(name)
        r = measureDelay(mic0, mic1)
        delays[name] = r
        print("  %-16s %8.2f %9.4f %8.4f %9.4f"
              % (name, r["N"], r["tau"], r["cc"], r["spread"]))

tauAir = np.array([delays[n]["tau"] for n in airFiles])
tauBreath = np.array([delays[n]["tau"] for n in breathFiles])
print("  воздух: tau = %.4f ± %.4f мс (n = %d)"
      % (tauAir.mean(), tauAir.std(ddof=1), len(tauAir)))
print("  выдох:  tau = %.4f ± %.4f мс (n = %d)"
      % (tauBreath.mean(), tauBreath.std(ddof=1), len(tauBreath)))

# ================ Шаг 4 (п4.6). Скорости звука и доли CO2 ================
cAir = L / (np.mean(tauAir) / 1000)
cBreath = L / (np.mean(tauBreath) / 1000)

co2Air = np.polyval(fitRoom, cAir)
co2Breath = np.polyval(fitWet, cBreath)

print("\nВоздух: c = %.1f м/с, CO2 = %.1f %%" % (cAir, co2Air))
print("Выдох:  c = %.1f м/с, CO2 = %.1f %%" % (cBreath, co2Breath))

co2AirAll = np.polyval(fitRoom, L / (tauAir / 1000))
co2BreathAll = np.polyval(fitWet, L / (tauBreath / 1000))
print("  по записям, воздух: CO2 = %s %% (среднее %.1f ± %.1f)"
      % (np.round(co2AirAll, 1), co2AirAll.mean(), co2AirAll.std(ddof=1)))
print("  по записям, выдох:  CO2 = %s %% (среднее %.1f ± %.1f)"
      % (np.round(co2BreathAll, 1), co2BreathAll.mean(), co2BreathAll.std(ddof=1)))

# ------------------------- Оценка точности (т4.4) -------------------------
slope = abs(fitRoom[0])   # % CO2 на 1 м/с
semAir = tauAir.std(ddof=1) / np.sqrt(len(tauAir))
semBreath = tauBreath.std(ddof=1) / np.sqrt(len(tauBreath))
print("\nОценка точности (по т4.4):")
print("  источник                       dtau, мс   dc, м/с   dCO2, %")
for label, dtau in (("один отсчёт АЦП", SAMPLE_MS),
                    ("среднее по опытам, SEM", semAir),
                    ("разброс по записям, 1σ", tauAir.std(ddof=1))):
    dc = cAir * dtau / np.mean(tauAir)
    print("  %-30s %8.4f %9.2f %9.2f" % (label, dtau, dc, dc * slope))
dcL = cAir * 0.002 / L
print("  %-30s %8s %9.2f %9.2f" % ("расстояние L ± 2 мм", "—", dcL, dcL * slope))
dcT = 0.6 * 0.5
print("  %-30s %8s %9.2f %9.2f" % ("температура ± 0,5 °C", "—", dcT, dcT * slope))

# разность «выдох - воздух»: ошибка L и T почти сокращается
dcDiff = L * np.sqrt((semAir / np.mean(tauAir) ** 2) ** 2 +
                     (semBreath / np.mean(tauBreath) ** 2) ** 2) * 1000
print("\n  разность tau(выдох) - tau(воздух) = %.4f мс"
      % (np.mean(tauBreath) - np.mean(tauAir)))
print("  разность скоростей dc = %.1f м/с -> dCO2 = %.2f %% (L и T сокращаются)"
      % (abs(cAir - cBreath), dcDiff * slope))
print("  CO2 в выдохе по разности скоростей = %.1f %%"
      % (abs(cAir - cBreath) * slope))

# ============== Шаг 5 (п4.5, п4.6). Итоговый график speed-of-sound.png ==============
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(co2Plot, speedRoomPlot, "-", lw=1.8, label="расчёт, влажность %.1f %%" % rh)
ax.plot(co2Plot, speedWetPlot, "-", lw=1.8, label="расчёт, влажность 100 %")

ax.plot(co2Air, cAir, "*", markersize=17, markeredgecolor="k", markeredgewidth=0.6,
        label="Воздух: %.1f м/с, %.1f %%" % (cAir, co2Air))
ax.plot(co2Breath, cBreath, "*", markersize=17, markeredgecolor="k", markeredgewidth=0.6,
        label="Выдох: %.1f м/с, %.1f %%" % (cBreath, co2Breath))

ax.set_xlabel("Доля CO₂ в сухом воздухе, %")
ax.set_ylabel("Скорость звука, м/с")
ax.set_title("Скорость звука во влажном воздухе с примесью CO₂\n"
             "t = %.1f °C, влажность аудитории %.1f %%" % (t, rh))
ax.set_xlim(0, 7)
ax.grid(alpha=0.4)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(plotsPath + "speed-of-sound.png", dpi=200)


# ---- Дополнительно: три этапа определения задержки (т4.4, рисунок A -> D) ----
def firstDip(x, i0, span=800):
    """Индекс первого заметного провала после начала импульса."""
    seg = x[i0:i0 + span]
    return i0 + int(np.argmin(seg))


def plotStages(name, fname):
    """Три графика по этапам т4.4 (рисунок A -> D):
       1) исходные записи — фон не вычтен, амплитуды не выровнены, сдвига нет;
       2) фон вычтен, амплитуды выровнены — видно, что импульсы разнесены на τ;
       3) крупный план начала импульса, совмещённого сдвигом на τ.

    Окно панели 2 широкое: оно должно вмещать оба импульса, иначе задержку
    не показать. Окно панели 3 узкое, только начало импульса: дальше идут
    отражения от концов трубы и дисперсия, из-за которых формы расходятся
    (т4.4, «Что ещё есть на записи»), а для задержки важно лишь начало.
    """
    raw0 = np.loadtxt(dataPath + name + "_0.txt").astype(float)
    raw1 = np.loadtxt(dataPath + name + "_1.txt").astype(float)

    # все величины считаем заново и согласованно, одним прогоном
    a = raw0 - np.mean(raw0[:200])
    b = raw1 - np.mean(raw1[:200])
    r = findDelay(raw0, raw1)
    N = r["N"]
    Ni = int(round(N))
    i0 = r["i0"]
    i1 = i0 + Ni                     # то же событие в микрофоне 1

    # высоты первых пиков (шаги 3-4 из т4.4): пик направлен вниз, берём модуль
    A0 = abs(a[firstDip(a, i0)])
    A1 = abs(b[firstDip(b, i1)])

    # ---- широкое окно: оба импульса целиком ----
    pre_w = 400
    post_w = Ni + 700
    sW = max(0, i0 - pre_w)
    winW = min(pre_w + post_w, len(a) - sW - Ni)
    tW = (np.arange(winW) + sW) * SAMPLE_MS
    tW1 = tW + MIC1_SHIFT_MS

    # ---- узкое окно: только начало импульса ----
    pre_n = 300
    post_n = 700
    sN = max(0, i0 - pre_n)
    winN = pre_n + post_n
    tN = (np.arange(winN) + sN) * SAMPLE_MS

    aWN = a[sW:sW + winW] / A0
    bWN = b[sW:sW + winW] / A1
    aNN = a[sN:sN + winN] / A0
    bNN = b[sN + Ni:sN + Ni + winN] / A1      # сдвинут на τ

    d0 = firstDip(a, i0) * SAMPLE_MS
    d1 = (firstDip(b, i1) + MIC1_SHIFT_MS) * SAMPLE_MS

    fig, axes = plt.subplots(3, 1, figsize=(10, 11.5))

    # --- 1. Исходные записи: как есть ---
    ax = axes[0]
    ax.plot(tW, raw0[sW:sW + winW], lw=1.1, label="микрофон 0")
    ax.plot(tW1, raw1[sW:sW + winW], lw=1.1, label="микрофон 1")
    ax.set_title("1. Исходные записи: фон не вычтен, амплитуды не выровнены, сдвига нет")
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("Код АЦП")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.4)

    # --- 2. Фон вычтен, амплитуды выровнены; сдвига ещё нет ---
    ax = axes[1]
    ax.plot(tW, aWN, lw=1.2, label="микрофон 0")
    ax.plot(tW1, bWN, lw=1.2, label="микрофон 1")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    yA = 0.55
    for xd in (d0, d1):
        ax.axvline(xd, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.annotate("", xy=(d1, yA), xytext=(d0, yA),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.4,
                                shrinkA=0, shrinkB=0))
    ax.text((d0 + d1) / 2, yA + 0.05,
            "τ = %.3f мс" % (d1 - d0), ha="center", va="bottom", fontsize=12)
    ax.set_ylim(min(aWN.min(), bWN.min()) - 0.15, max(aWN.max(), bWN.max()) + 0.35)
    ax.set_title("2. Фон вычтен, амплитуды выровнены по первому пику: "
                 "импульсы разнесены на τ")
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("Сигнал / высота первого пика")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.4)

    # --- 3. Крупный план: сигналы совмещены сдвигом на τ ---
    ax = axes[2]
    ax.plot(tN, aNN, lw=1.8, label="микрофон 0")
    ax.plot(tN, bNN, lw=1.3, ls="--", label="микрофон 1, сдвинут на τ")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(d0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylim(min(aNN.min(), bNN.min()) - 0.12, max(aNN.max(), bNN.max()) + 0.12)
    ax.set_title("3. Начало импульса после сдвига на τ = %.3f мс (N = %.1f отсчёта): "
                 "кривые совпадают" % (r["tau"], r["N"]))
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("Сигнал / высота первого пика")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.4)

    fig.suptitle("Определение задержки по записи %s" % name, fontsize=13)
    fig.tight_layout()
    fig.savefig(plotsPath + fname, dpi=200)
    plt.close(fig)


plotStages("AIR7hlop", "delay-air.png")
plotStages("breath1", "delay-breath.png")

# ------------------------ Сводная таблица для отчёта ------------------------
with open(plotsPath + "results.md", "w", encoding="utf-8") as f:
    f.write("# Результаты обработки (п4.5, п4.6)\n\n")
    f.write("Условия опыта: t = %.1f °C, относительная влажность %.1f %%.\n\n" % (t, rh))
    f.write("## Прямые (x — скорость звука, м/с; y — доля CO₂, %)\n\n")
    f.write("| Влажность | Наклон, %/(м/с) | Свободный член |\n| --- | --- | --- |\n")
    f.write("| %.1f %% (аудитория) | %.4f | %.3f |\n" % (rh, fitRoom[0], fitRoom[1]))
    f.write("| 100 %% (выдох) | %.4f | %.3f |\n\n" % (fitWet[0], fitWet[1]))
    f.write("Расчётная скорость звука при 0 %% CO₂: %.2f м/с (влажность аудитории), "
            "%.2f м/с (влажность 100 %%).\n\n" % (speedRoom[0], speedWet[0]))
    f.write("## Задержки по записям\n\n")
    f.write("| Запись | N, отсчётов | τ, мс | Корреляция | Разброс по сетке, мс |\n")
    f.write("| --- | --- | --- | --- | --- |\n")
    for name in airFiles + breathFiles:
        r = delays[name]
        f.write("| %s | %.2f | %.4f | %.4f | %.4f |\n"
                % (name, r["N"], r["tau"], r["cc"], r["spread"]))
    f.write("\n## Итог\n\n")
    f.write("Доля CO₂ по каждой записи: воздух %s %%, выдох %s %%.\n\n"
            % (np.round(co2AirAll, 1), np.round(co2BreathAll, 1)))
    f.write("| Группа | τ, мс | c, м/с | CO₂, % |\n| --- | --- | --- | --- |\n")
    f.write("| Атмосферный воздух | %.4f ± %.4f | %.1f | %.1f |\n"
            % (tauAir.mean(), tauAir.std(ddof=1), cAir, co2Air))
    f.write("| Выдох | %.4f ± %.4f | %.1f | %.1f |\n"
            % (tauBreath.mean(), tauBreath.std(ddof=1), cBreath, co2Breath))
    f.write("\nПогрешность доли CO₂: ±%.1f %% (разброс по записям), "
            "±%.1f %% (среднее по опытам).\n"
            % (abs(cAir * tauAir.std(ddof=1) / tauAir.mean()) * slope,
               abs(cAir * semAir / tauAir.mean()) * slope))

print("\nГотово:")
print("  %sspeed-of-sound.png — итоговый график" % plotsPath)
print("  %sdelay-air.png, %sdelay-breath.png — совмещение сигналов" % (plotsPath, plotsPath))
print("  %sresults.md — сводная таблица" % plotsPath)

if "agg" not in matplotlib.get_backend().lower():
    plt.show()
