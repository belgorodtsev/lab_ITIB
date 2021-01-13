'''
Лабораторная работа No 2
Применение однослойной нейронной сети с линейной
функцией активации для прогнозирования временных рядов
Вариант 2.
'''
from math import *
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy
import copy
import random


def get_func(t):
    '''
    Функция возвращает значении функции
    :param t: точка
    :param return: значение функции в этой точке
    '''
    return 0.5 * cos(0.5 * t) - 0.5


def line_plot(x_data, y_data, x_value, y_value, x_label="", y_label="", title=""):
    '''
    Функция для построения графика
    :param x_data: точки на всём отрезке от [a, 2b-a]
    :param x_data: значение функции на всём отрезке
    :param x_value: точки на полуинтервале от (b, 2b-a]
    :param x_value: предсказанные значения функции на этом полуинтервале
    '''
    _, ax = plt.subplots()
    ax.plot(x_data, y_data, label='График функции', lw=2, color='#539caf', alpha=0.5)
    ax.scatter(x_data[:(len(x_data) // 2 + 1)],
               y_data[:(len(y_data) // 2 + 1)],
               marker='o', s=10, c='b', alpha=0.3)
    ax.plot(x_value, y_value, label='График прогноза', marker='o', lw=1, color='r', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.show()


def learning(vec_weights, vec_x, size_p, learning_rate, era):
    '''
    Функция обучения НС
    :param vec_weights: началные веса НС
    :param vec_x: значение точек на всём отрезке
    :param size_p: длина окна
    :param learning_rate: норма обучения
    :param era: количество эпох
    :param return: конечные веса НС, последний столбец обучающей выборки
    '''
    counter_era = 0
    # векторы-столбцы обучающей выборки
    vec_column = [vec_x[i:i + size_p] for i in range(0, len(vec_x) - size_p + 1, 1)]
    table = PrettyTable(['Era', 'Weight'])
    for _ in range(era):
        for i in range(len(vec_column) - 1):  # столбцы
            net = vec_weights[0]  # вычисляем прогнозируемое значение
            for k in range(size_p):
                net += vec_weights[k + 1] * vec_column[i][k]
            delta = vec_column[i + 1][size_p - 1] - net  # ошибка прогноза
            vec_weights[0] += learning_rate * delta  # новый вес
            for k in range(size_p):
                vec_weights[k + 1] += learning_rate * delta * vec_column[i][k]
        counter_era += 1
        table.add_row([counter_era, copy.copy(vec_weights)])
    # print(table)
    return vec_weights, vec_column[len(vec_column) - 1]


def forecasting(vec_weights, vec_window, size_p, target_values):
    '''
    Функция для предсказания значения функции
    :param vec_weights: конечные веса НС
    :param vec_window: последний столбец обучающей выборки
    :param size_p: длина окна
    :param target_values: целевые значения функции
    :param return: предсказанные значения функции, ошибка
    '''
    vec_new_points = vec_window
    error = 0
    for i in range(len(target_values)):
        net = vec_weights[0]
        for k in range(size_p):
            net += vec_weights[k + 1] * vec_new_points[len(vec_new_points) - size_p + k]
        error += (target_values[i] - net) ** 2
        vec_new_points.append(net)
    return vec_new_points, sqrt(error)


def error_dependency(vec_points, target_values):
    _, ax = plt.subplots()
    era = 100000
    learning_rate = 1
    vec_y = list()
    vec_x = list()
    for size_p in range(4, 20):
        print(size_p)
        vec_weights = [0 for _ in range(size_p + 1)]
        weights, vec_window = learning(vec_weights,
                                       [get_func(t) for t in vec_points],
                                       size_p,
                                       round(learning_rate, 3),
                                       era)
        predicted_val, eps_val = forecasting(weights, vec_window, size_p, target_values)
        vec_y.append(eps_val)
        vec_x.append(size_p)
    ax.plot(vec_x, vec_y)
    ax.scatter(vec_x, vec_y, marker='o', s=5, alpha=0.6)
    ax.set_title('Зависимость ошибки от длины окна. M = 100000, n = 1')
    ax.set_xlabel('p')
    ax.set_ylabel('e')
    plt.grid()
    plt.savefig('eP.png')


def learning_analysis(vec_points, target_values):
    '''
    Функция для нахождения оптимальных параметров обучения
    :param vec_points: точки на отрезке [a, b]
    :param target_values: значения функции на отрезке (b, 2b - a]
    :param return: результаты прогона
    '''
    eps = 1
    data = list()
    table = PrettyTable(['Window', 'Era', 'Learning rate', 'Error'])
    for size_p in range(4, 20):
        print(size_p)
        for era in range(10000, 110000, 10000):
            for learning_rate in numpy.arange(0.1, 1.1, 0.1):
                vec_weights = [0 for _ in range(size_p + 1)]
                weights, vec_window = learning(vec_weights,
                                               [get_func(t) for t in vec_points],
                                               size_p,
                                               round(learning_rate, 3),
                                               era)
                predicted_val, eps_val = forecasting(weights, vec_window, size_p, target_values)
                if eps_val < eps:
                    eps = eps_val
                    data = [size_p, era, learning_rate, eps]
                table.add_row([size_p, era, round(learning_rate, 3), eps_val])
    table_txt = table.get_string()
    with open('output.txt', 'w') as file:
        file.write(table_txt + '\n')
        file.write('Минимальные параметры' + str(data))
    file.close()
    return data


if __name__ == "__main__":
    a = -5
    b = 5
    c = 2 * b - a

    N = 20
    step = (abs(a) + abs(b)) / (N - 1)
    # точки на отрезке [a,b]
    points = [round(float(i), 3) for i in numpy.arange(a, b + step, step)]  # исходные точки
    # точки на отрезке [a, 2b - a]
    points_final = [round(float(i), 3) for i in numpy.arange(a, c + step, step)]  # весь график

    target_values = [get_func(t) for t in points_final[20:]]

    p = int(input('Введите длину скользящего окна:'))
    nu = float(input('Введите норму обучения:'))
    M = int(input('Введите колличество эпох обучения:'))

    weights = [0 for _ in range(p + 1)]

    weights, points_window = learning(weights, [get_func(t) for t in points], p, nu, M)
    print('Веса НС', weights)

    predicted_value, eps = forecasting(weights, points_window, p, target_values)
    print('Ошибка', eps)

    # точки для которых прогнозируется функция
    predicted_point = [points_final[i] for i in range(len(points) - p, len(points_final))]

    line_plot(points_final, [get_func(t) for t in points_final], predicted_point, predicted_value, 't', 'X(t)',
              'p = ' + str(p) + ', n = ' + str(nu) + ', M = ' + str(M) + ', eps = ' + str(round(eps, 4)))
