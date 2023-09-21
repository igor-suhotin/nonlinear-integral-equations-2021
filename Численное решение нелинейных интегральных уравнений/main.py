# -*- coding: utf-8 -*-

# Либы

import numpy as np
import matplotlib.pyplot as plt
from num_methods import volt_II_nonlin_solve, fred_II_nonlin_solve

#%% Гиперпараметры

h = 1e-1         # шаг сетки
abstol = 1e-10   # абсолютная погрешность

#%% Параметры для уравнения (1) Вольтерра 2 рода

yt = np.vectorize(lambda x: 1)                                # истинное решение 
K = np.vectorize(lambda x, s, y: np.exp(-(x - s)) * y ** 2)   # ядро
der = np.vectorize(lambda x, s, y: 2 * np.exp(-(x - s)) * y)  # ЧАСТНАЯ производная от ядра по y
f = np.vectorize(lambda x: np.exp(-x))                        # правая часть
xlim = (0, 1)                                                 # отрезок интегрирования

#%% Параметры для уравнения (2) Вольтерра 2 рода

yt = np.vectorize(lambda x: x)                                # истинное решение 
K = np.vectorize(lambda x, s, y: (1 + y ** 2) / (1 + s ** 2)) # ядро
der = np.vectorize(lambda x, s, y: 2 * y / (1 + s ** 2))      # ЧАСТНАЯ производная от ядра по y
f = np.vectorize(lambda x: 0.)                                # правая часть
xlim = (0, 2)                                                 # отрезок интегрирования

#%% Параметры для уравнения (1) Фредгольма 2 рода

yt = np.vectorize(lambda x: np.exp(-(x ** 2 / 2)))                               # истинное решение 
K = np.vectorize(lambda x, s, y: x * s * y ** 2)                                 # ядро
der = np.vectorize(lambda x, s, y: 2 * x * s * y)                                # ЧАСТНАЯ производная от ядра по y
l = 0.125                                                                        # лямбда
f = np.vectorize(lambda x: np.exp(-(x ** 2 / 2)) + x * (1 / np.e - 1) / 16)      # правая часть
xlim = (0, 1)                                                                    # отрезок интегрирования

#%% Параметры для уравнения (2) Фредгольма 2 рода

yt = np.vectorize(lambda x: np.cos(x))                                           # истинное решение
K = np.vectorize(lambda x, s, y: np.cos(x) * np.sin(s) * np.log(1 + y))          # ядро
der = np.vectorize(lambda x, s, y: np.cos(x) * np.sin(s) / (1 + y))              # ЧАСТНАЯ производная от ядра по y
l = 0.125                                                                        # лямбда
f = np.vectorize(lambda x: 0.125 * np.cos(x) * (10 - np.log(2) - 1 / np.log(2))) # правая часть
xlim = (0, np.pi / 2)                                                            # отрезок интегрирования

#%% rectange + quad_simple для Вольтерра

x, y = volt_II_nonlin_solve(K, f, 
                            xlim, 
                            h=h, 
                            method_int="rectangle", method_solve="quad_simple", 
                            abstol=abstol)
print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% trapeze + quad_simple для Вольтерра

x, y = volt_II_nonlin_solve(K, f, 
                            xlim, 
                            h=h, 
                            method_int="trapeze", method_solve="quad_simple", 
                            abstol=abstol)
print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% rectangle + quad_newt для Вольтерра

x, y = volt_II_nonlin_solve(K, f, 
                            xlim, 
                            h=h, 
                            method_int="rectangle", method_solve="quad_newt",
                            der=der,
                            abstol=abstol)
print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% trapeze + quad_newt для Вольтерра

x, y = volt_II_nonlin_solve(K, f, 
                            xlim,
                            h=h, 
                            method_int="trapeze", method_solve="quad_newt",
                            der=der,
                            abstol=abstol)
print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% rectangle + iter_zelder для Вольтерра

x, y = volt_II_nonlin_solve(K, f, 
                           xlim, 
                           h=h, 
                           method_int="rectangle", method_solve="iter_zelder", 
                           abstol=1e-10)

print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% trapeze + iter_simple для Вольтерра

x, y = volt_II_nonlin_solve(K, f, 
                           xlim, 
                           h=h, 
                           method_int="simpson", method_solve="iter_simple", 
                           abstol=abstol)

print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% simpson + quad_newt для Фредгольма

x, y = fred_II_nonlin_solve(K, f, l,
                            xlim,
                            h=h,
                            method_int="simpson", method_solve="quad_newt",
                            der=der,
                            abstol=abstol)
print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% simpson + quad_simple для Фредгольма

x, y = fred_II_nonlin_solve(K, f, l,
                            xlim,
                            h=h,
                            method_int="simpson", method_solve="quad_simple",
                            abstol=abstol)
print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% simpson + iter_simple для Фредгольма

x, y = fred_II_nonlin_solve(K, f, l,
                            xlim,
                            h=h,
                            method_int="simpson", method_solve="iter_simple",
                            abstol=abstol)
print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% rectangle + iter_zelder для Фредгольма

x, y = fred_II_nonlin_solve(K, f, l,
                            xlim,
                            h=h,
                            method_int="rectangle", method_solve="iter_zelder",
                            abstol=1e-10)
print("x\ty\tyt\terr")
print(np.concatenate((x.reshape((-1, 1)), 
                      y.reshape((-1, 1)),
                      yt(x).reshape((-1, 1)),
                      np.abs(yt(x) - y).reshape((-1, 1))),
                     axis=1))

#%% График для уравнения Вольтерра (1)

plt.figure(figsize=(7, 7))
plt.plot(x, yt(x), "-", label="Истинное решение")
plt.plot(x, y, marker='o', linestyle='dashed', label="Численное решение")
plt.xlim((x[0], x[-1])) 
plt.ylim((0.5, 1.5))
plt.xlabel('x', fontsize=12, color='blue')
plt.ylabel('y', fontsize=12, color='blue')
plt.legend()
plt.grid(True)

#%% График для уравнения Вольтерра (2)

plt.figure(figsize=(7, 7))
plt.plot(x, yt(x), "-", label="Истиное решение")
plt.plot(x, y, marker='o', linestyle='dashed', label="Численное решение")
plt.xlim((x[0], x[-1])) 
plt.ylim((-1, 3))
plt.xlabel('x', fontsize=12, color='blue')
plt.ylabel('y', fontsize=12, color='blue')
plt.legend()
plt.grid(True)

#%% График для уравнения Фредгольма (1)

plt.figure(figsize=(7, 7))
plt.plot(x, yt(x), "-", label="Истиное решение")
plt.plot(x, y, marker='o', linestyle='dashed', label="Численное решение")
plt.xlim((x[0], x[-1]))
plt.ylim((0.5, 1.1))
plt.xlabel('x', fontsize=12, color='blue')
plt.ylabel('y', fontsize=12, color='blue')
plt.legend()
plt.grid(True)

#%% График для уравнения Фредгольма (2)

plt.figure(figsize=(7, 7))
plt.plot(x, yt(x), "-", label="Истиное решение")
plt.plot(x, y, marker='o', linestyle='dashed', label="Численное решение")
plt.xlim((x[0], x[-1]))
plt.ylim((0.0, 1.1))
plt.xlabel('x', fontsize=12, color='blue')
plt.ylabel('y', fontsize=12, color='blue')
plt.legend()
plt.grid(True)
