'''
Лабораторная работа No 1
Исследование однослойных нейронных сетей на примере моделирования булевых выражений.
Цель: Исследовать функционирование простейшей нейронной сети (НС) на базе нейрона с
нелинейной функцией активации и ее обучение по правилу Видроу-Хоффа.
Вариант 2.
'''
from math import *
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import copy
import itertools

'''     
Функция возвращает значения БФ на заданом ей наборе   
:param X: набор переменных значения БФ
:param return: значения БФ
'''


def boolean_func(X):
    return (((not X[2]) or X[3]) and (not X[0])) or X[1]  # второй вариант


# return (not (X[0] and X[1])) and X[2] and X[3]  # метода


'''
Функция инциализирует необходимые для расчётов компоненты, выводит таблицу истинности
:param n: количество переменных
:param return: F - значения БФ, W - начальные весовые коэффициенты
'''


def init_F(n):
    X = bin_generation(n)
    F = get_F(X)
    t = PrettyTable(['X', 'F'])
    for x, f in zip(X, F):
        t.add_row([x, f])
    print(t)
    return F


'''
Функция генерирует наборы X
:param n: количество переменных
:param return: X - список наборов
'''


def bin_generation(n):
    X = list()
    for i in range(0, 2 ** n):
        X.append(IntToByte(i, n))
    return X


def IntToByte(x, n):
    v = [0 for _ in range(n)]
    i = n - 1;
    while x > 0:
        v[i] = x % 2
        x = x // 2
        i = i - 1
    return v


'''     
Функция для построения графика E(k)
'''


def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    ax.plot(x_data, y_data, lw=2, marker='o', color='#539caf', alpha=0.5)
    # ax.scatter(x_data, y_data, s=10, color='#539caf', alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.grid()
    plt.show()


'''     
Функция возвращает значения БФ на заданных ей наборах переменных     
:param X: наборы переменных значения БФ
:param return: значения БФ F
'''


def get_F(X):
    F = list()
    for x in X:
        F.append(int(boolean_func(x)))
    return F


'''     
Пороговая функция активации   
:param net: сетевой вход
:param return: значение функции fnet, и её производной dfnet
'''


def get_FA1(net):
    return 1 if net >= 0 else 0, 1


'''
Сигмоидальная функция активации   
:param net: сетевой вход
:param return: значение функции fnet, и её производной dfnet
'''


def get_FA2(net):
    fnet = 1 / (1 + exp(-net))
    dfnet = fnet * (1 - fnet)
    return 1 if fnet >= 0.5 else 0, dfnet


'''     
Функция возвращает значения Y реальный выход НС    
:param X: наборы переменных значения БФ
:param W: веса НС
:param func: ФА
:param n: количество переменных
:param return: значения Y - реальный выход
'''


def get_Y(X, W, func):
    Y = [0 for _ in range(len(X))]
    f = 0
    for j in range(len(X)):
        net = W[0]  # считаем net
        for i in range(0, len(X[j])):
            net += X[j][i] * W[i + 1]
        if func == '1':
            f, df = get_FA1(net)
        elif func == '2':
            f, df = get_FA2(net)
        Y[j] = f  # реальный выход
    return Y


'''     
Функция считает ошибку (расстояние Хемминга)    
:param Y: реальный выход
:param F: целевой выход
:param return: значения Y - реальный выход
'''


def hamming_distance(F, Y):
    return sum(f != y for f, y in zip(F, Y))


'''     
Функция производит все рассчёты обучения  
:param X: наборы переменных значения БФ
:param W: веса НС
:param Y: реальный выход
:param F: целевой выход
:param func: ФА
:param nu: норма обучения
:param n: количество переменных
:param setSize: количество наборов для обучения
:param return: значения Y - реальный выход, W - веса НС
'''


def calculate(X, W, F, func, nu, numberSets, x0=1):
    for l in range(0, numberSets):
        net = W[0]  # считаем net
        for i in range(0, len(X[l])):
            net += X[l][i] * W[i + 1]
        if func == '1':
            f, df = get_FA1(net)
        elif func == '2':
            f, df = get_FA2(net)
        d = F[l] - f  # F[l] = t правило Видроу - Хоффа
        W[0] += nu * d * df * x0  # новый вес
        for i in range(0, len(X[l])):
            W[i + 1] += nu * d * df * X[l][i]
    return W, get_Y(X, W, func)


'''     
Функция обучения на всех наборах
:param F: реальный выход
:param func: ФА
:param nu: норма обучения
:param n: количество переменных
:param return: значения Y - реальный выход, W - веса НС, K,E - номера эпох и ошибки (для графика)
'''


def learning(F, nu, func, n):
    k = 0  # эпоха
    X = bin_generation(n)
    W = [0 for _ in range(n + 1)]
    Y = get_Y(X, W, func)
    error = hamming_distance(F, Y)  # квадратична ошибка
    t = PrettyTable(['K', 'W', 'Y', 'E'])
    t.add_row([k, copy.copy(W), copy.copy(Y), error])
    K = [k]
    E = [error]
    while error != 0:
        W, Y = calculate(X, W, F, func, nu, len(X))
        error = hamming_distance(F, Y)
        E.append(error)
        k += 1
        K.append(k)
        t.add_row([k, copy.copy(W), copy.copy(Y), error])
    print(t)
    return K, W, Y, E


'''     
Функция обучения на выборочных наборах
:param F: реальный выход
:param func: ФА
:param nu: норма обучения
:param numberSets: количество наборов для обучения
:param n: количество переменных
:param return: значения Y - реальный выход, W - веса НС, K,E - номера эпох и ошибки (для графика)
'''


def selective_lerning(F, nu, func, n):
    k = 0  # эпоха
    X = bin_generation(n)
    W = [0 for _ in range(n + 1)]
    numberSets = 2  # начнём перебор с 2 наборов
    while True:
        for setX, setF in zip(itertools.combinations(X, numberSets), itertools.combinations(F, numberSets)):
            setY = [1 for _ in range(numberSets)]
            e = hamming_distance(setF, setY)  # квадратична ошибка
            t = PrettyTable(['K', 'W', 'Y', 'E'])
            t.add_row([k, copy.copy(W), copy.copy(setY), e])
            # print(k, copy.copy(W), copy.copy(setY), e)
            K = [k]
            E = [e]
            while e != 0 and k < 100:
                W, setY = calculate(setX, W, setF, func, nu, numberSets)
                setY = get_Y(setX, W, func)
                e = hamming_distance(setF, setY)
                E.append(e)
                k += 1
                K.append(k)
                t.add_row([k, copy.copy(W), copy.copy(setY), e])
            Y = get_Y(X, W, func)  # получаем Y по нашим весам и считаем ошибку
            if hamming_distance(F, Y) == 0:
                print('Удалось обучить на', numberSets, 'наборах')
                for i in range(numberSets):
                    print('X' + str(i + 1), '=', setX[i], end=' ')
                print()
                print(t)
                return K, W, Y, E
            else:
                k = 0
                W = [0 for _ in range(n + 1)]
                K.clear()
                E.clear()
                t.clear()
        numberSets += 1


if __name__ == "__main__":
    F = init_F(n=4)
    print('Задание 1')
    function = input('Введите ФА ("1" - пороговая ФА, "2" - сигмоидальная ФА):')
    nu = float(input('Введите норму обучения:'))
    K, W, Y, E = learning(F, nu, function, n=4)
    lineplot(K, E, "Error E", "Era K", "E(k)")
    print('Задание 2')
    function = input('Введите ФА ("1" - пороговая ФА, "2" - сигмоидальная ФА):')
    nu = float(input('Введите норму обучения:'))
    K, W, Y, E = selective_lerning(F, nu, function, n=4)
    lineplot(K, E, "Error E", "Era K", "E(k)")
