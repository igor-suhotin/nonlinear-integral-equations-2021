# -*- coding: utf-8 -*-

import numpy as np

def LUP(A): 
    """
    Вычисляет LUP-разложение для матрицы
    
    Параметры:
    A - входная матрица
    
    Возвращает: 
    LU - матрица, такая, что ненулевые элементы L располагаются ниже главной диагонали LU,
         а ненулевые элементы U - не ниже главной диагонали LU
    pi - вектор перестановок
    """
    LU = np.copy(A)
    n = LU.shape[0]
    pi = np.arange(n)
    for k in range(n):
        k_ = np.argmax(np.abs(LU[k:, k])) + k
        if k != k_:
            pi[k], pi[k_] = pi[k_], pi[k]
            LU[k, :], LU[k_, :] = LU[k_, :], LU[k, :]
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            for j in range(k + 1, n):
                LU[i, j] -= LU[i, k] * LU[k, j]
    return LU, pi

def LUP_solve(LU, pi, b): 
    """
    Решает линейную систему Ax=b через LUP разложение матрицы A
    
    Параметры:
    LU - матрица, такая, что ненулевые элементы L располагаются ниже главной диагонали LU,
         а ненулевые элементы U - не ниже главной диагонали LU
    pi - вектор перестановок
    b - правая часть
    
    Возвращает:
    x - решение системы Ax=b
    """
    n = LU.shape[0]
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[pi[i]] - np.dot(LU[i, :i], y[:i]) # прямой ход
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LU[i, i + 1:], x[i + 1:])) / LU[i, i] # обратный ход
    return x

def eq_solve(f, x0, method="iter", der=None, abstol = 1e-10):
    """
    Решает трансцендентное уравнение f(x)=0 или x=f(x)
    
    Параметры:
    f      - функция
    x0     - начальное приближение
    method =
        "iter" - решение x=f(x) методом итераций
        "newt" - решение f(x)=0 метод Ньютона
    der    - частная производная f(x) для method="iter"
    abstol - абсолютная погрешность вычислений
        
    Возвращает:
    x - решение трансцендентного уравнения f(x)=0 или x=f(x)
    """
    x = x0
    x_ = np.Inf
    if method == "iter": # Метод итераций
        while np.abs(x - x_) >= abstol:
            x_ = x
            x = f(x)
    elif method == "newt": # метод Ньютона
        while np.abs(x - x_) >= abstol:
            x_ = x
            x = x - f(x) / der(x)
    return x

# с несколькими неизвестными корнями, т.е. систему
def sys_solve(F, x0, method="iter", Jac=None, ord=None, abstol = 1e-10):
    """
    Решает систему трансцендентных уравнений F(x)=0 или x=F(x), где
    F = [F1, F2, ..., Fn]
    
    Параметры:
    F      - функция n переменных
    x0     - начальное приближение
    method =
        "iter"    - решение x=F(x) методом итераций
        "zelder"  - решение x=F(x) методом Зейделя
        "newt"    - решение F(x)=0 метод Ньютона (требуется Jac)
    Jac    - Якобиан отображения F
    ord    - вид нормы
    abstol - абсолютная погрешность вычислений
        
    Возвращает:
    x - решение трансцендентного уравнения f(x)=0 или x=f(x)
    """
    x = x0
    x_ = np.Inf * np.ones_like(x)
    if method == "iter": # Метод итераций
        while np.linalg.norm(x - x_, ord=ord) >=abstol:
            x_ = x
            x = F(x)
    elif method == "zelder": # Метод Зейделя
        while np.linalg.norm(x - x_, ord=ord) >=abstol:
            x_ = x
            for i in range(len(x)):
                x[i] = F(x)[i]
    elif method == "newt": # метод Ньютона
        while np.linalg.norm(x - x_, ord=ord) >=abstol:
            x_ = x
            A = Jac(x)
            b = -F(x)
            LU, pi = LUP(A)
            x += LUP_solve(LU, pi, b)
    return x

def volt_II_nonlin_solve(K, f, 
                         xlim, 
                         h=1e-2, 
                         method_int="rectangle", method_solve="iter",
                         der=None,
                         ord=None,
                         abstol = 1e-10):
    """
    Решает нелинейное уравнение Вольтерра 2 рода
    
    Параметры:
    K      - ядро: функция 3 переменных x, s, y
    f      - правая часть
    xlim   - отрезок интегрирования
    h      - шаг сетки
    method_int =
        "rectangle" - инициализация массива коэффициентов перед ядром 
                      по формуле прямоугольников
        "trapeze"   - ... по формуле трапеций
        "simpson"   - ... по формуле Симпсона (парабол)
    method_solve =
        "quad_simple" - решение методом квадратур, в процессе решается одномерное
                        трансцендентных уравнение методом простых итераций
        "quad_newt"   - решение методом квадратур, в процессе решается одномерное
                        трансцендентных уравнение методом Ньютона
                        (требуется der)
        "iter_simple" - решение методом итераций
        "iter_zelder" - решение методом Зейделя
    der    - частная производная ядра по y
    ord    - вид нормы
    abstol - абсолютная погрешность вычислений
        
    Возвращает:
    x - массив аргументов
    y - массив значений
    """
    x = np.arange(xlim[0], xlim[1] + h, h, dtype=np.double)
    n = len(x) # длина сетки
    y = np.zeros(shape=n, dtype=np.double)
    
    # инициализация массива коэффициентов перед ядром
    if method_int == "rectangle": 
        A = np.concatenate(([0], h * np.ones(n - 2, dtype=np.double), [0]))  
    elif method_int == "trapeze":
        A = np.concatenate(([h / 2], h * np.ones(n - 2, dtype=np.double), [h / 2]))
    elif method_int == "simpson":
        A = np.concatenate(([h / 3], 
                            [4 * h / 3, 2 * h / 3] * (n - 2 >> 1), 
                            [4 * h / 3] if n & 1 == 1 else [],  
                            [h / 3]))
        
    method, kind = method_solve.split(sep="_")
    
    # решение, полученное разными методами
    
    if method == "quad": # квадратурные методы
        
        y[0] = f(x[0])
        
        if kind == "simple": # решением нелинейного уравнения методом итераций
            for i in range(1, n):
                y[i] = eq_solve(lambda yi: A[i] * K(x[i], x[i], yi) + f(x[i]) + np.dot(A[:i], K(x[i], x[:i], y[:i])), 
                                x0 = f(x[i]), method="iter", 
                                abstol=abstol)
        elif kind == "newt": # решением нелинейного уравнения методом Ньютона
            for i in range(1, n):
                y[i] = eq_solve(lambda yi: yi - A[i] * K(x[i], x[i], yi) - f(x[i]) - np.dot(A[:i], K(x[i], x[:i], y[:i])), 
                                x0 = f(x[i]), method="newt", 
                                der=lambda yi: 1 - A[i] * der(x[i], x[i], yi),
                                abstol=abstol)
        
    elif method == "iter": # итерационные методы
        
        y = f(x) # начальное приближение
        y_ = np.Inf * np.ones_like(y)
        while np.linalg.norm(y - y_, ord=ord) >= abstol:
            y_ = np.copy(y) # сохраняем старый массив
            y[0] = f(x[0])
            for i in range(1, n):
                y[i] = f(x[i]) + np.dot(A[:i + 1], K(x[i], x[:i + 1], y_[:i + 1] if kind == "iter" else y[:i + 1]))
    return x, y

# Фредгольма 2 рода
def fred_II_nonlin_solve(K, f, l,
                         xlim, 
                         h=1e-2, 
                         method_int="rectangle", method_solve="quad_newt",
                         der=None,
                         ord=None,
                         abstol = 1e-10):
    """
    Решает нелинейное уравнение Фредгольма 2 рода
    
    Параметры:
    K      - ядро: функция 3 переменных x, s, y
    f      - правая часть
    l      - lambda
    xlim   - отрезок интегрирования
    h      - шаг сетки
    method_int =
        "rectangle" - инициализация массива коэффициентов перед ядром 
                      по формуле прямоугольников
        "trapeze"   - ... по формуле трапеций
        "simpson"   - ... по формуле Симпсона (парабол)
    method_solve =
        "quad_simple" - решение методом квадратур, в процессе решается система
                        трансцендентных уравнение методом простых итераций
        "quad_zelder" - решение методом квадратур, в процессе решается система
                        трансцендентных уравнение методом Зейделя
        "quad_newt"   - решение методом квадратур, в процессе решается система
                        трансцендентных уравнение методом Ньютона
                        (требуется der)
        "iter_simple" - решение методом итераций
        "iter_zelder" - решение методом Зейделя
    der    - частная производная ядра по y
    ord    - вид нормы
    abstol - абсолютная погрешность вычислений
        
    Возвращает:
    x - массив аргументов
    y - массив значений
    """
    x = np.arange(xlim[0], xlim[1] + h, h, dtype=np.double)
    n = len(x) # длина сетки
    y = np.zeros_like(x)
    
    # инициализация массива коэффициентов перед ядром
    if method_int == "rectangle": 
        A = np.concatenate(([0], h * np.ones(n - 2, dtype=np.double), [0]))  
    elif method_int == "trapeze":
        A = np.concatenate(([h / 2], h * np.ones(n - 2, dtype=np.double), [h / 2]))
    elif method_int == "simpson":
        A = np.concatenate(([h / 3], 
                            [4 * h / 3, 2 * h / 3] * (n - 2 >> 1), 
                            [4 * h / 3] if n & 1 == 1 else [],  
                            [h / 3]))
        
    method, kind = method_solve.split(sep="_")
    
    # решение, полученное разными методами
    
    if method == "quad": # квадратурные методы
        
        if kind == "simple" or kind == "zelder": # простые итерации или Зейдель
            
            # инициализация вектор-функции x = F(x)
            F = lambda yi: np.array([l * np.dot(A, K(x[i], x, yi)) + f(x[i]) for i in range(n)])
            
            # решение нелинейной системы уравнений итерационными методами
            y = sys_solve(F, x0 = f(x), 
                          method=kind, 
                          ord=ord,
                          abstol=abstol)
            
        elif kind == "newt": # Ньютон
            
            # инициализация вектор-функции F(x) = 0
            F = lambda yi: np.array([yi[i] - l * np.dot(A, K(x[i], x, yi)) - f(x[i]) for i in range(n)])
            
            # инициализация Якобиана F(x)
            Jac = lambda yi: np.array([[1 - l * A[i] * der(x[i], x[i], yi[i]) if i == j else 
                                      - l * A[j] * der(x[i], x[j], yi[j]) 
                                       for j in range(n)] 
                                      for i in range(n)])
            
            # решение нелинейной системы уравнений методом Ньютона
            y = sys_solve(F, x0 = f(x), 
                          method="newt", 
                          ord=ord,
                          Jac=Jac,
                          abstol=abstol)
            
    elif method == "iter": # итерационные методы
        
        y = f(x) # начальное приближение
        y_ = np.Inf * np.ones_like(y)
        
        while np.linalg.norm(y - y_, ord=ord) >= abstol:
            y_ = np.copy(y) # сохраняем старый массив
            for i in range(n):
                y[i] = f(x[i]) + l * np.dot(A, K(x[i], x, y_ if kind == "simple" else y))
    return x, y
