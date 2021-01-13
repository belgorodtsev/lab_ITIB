import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools
from prettytable import PrettyTable


def line_plot(x_data, y_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    ax.plot(x_data, y_data, lw=2, marker='o', color='#539caf', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.grid()
    plt.show()


def boolean_func(vec_x):
    '''
    Функция возвращает значения БФ на заданом ей наборе
    :param vec_x: набор переменных значения БФ
    :param return: значения БФ
    '''
    return (((not vec_x[2]) or vec_x[3]) and (not vec_x[0])) or vec_x[1]  # второй вариант
    # return (not (vec_x[0] and vec_x[1])) and vec_x[2] and vec_x[3]  # метода


def hamming_distance(vec_f, vec_y):
    return sum(f != y for f, y in zip(vec_f, vec_y))


class TruthTable:
    '''
    Класс, который генерирует таблицу истиности,
    инициализирует целевую функцию и центры RBF нейронов
    '''

    def __init__(self, n: int):
        self.number_variables = n

    def init_parameters(self):
        vec_x = self.bin_generation()
        vec_f = list()
        for x in vec_x:
            vec_f.append(int(boolean_func(x)))
        vec_j1 = list()
        vec_j0 = list()
        t = PrettyTable(['X', 'F'])
        for x, f in zip(vec_x, vec_f):
            t.add_row([x, f])
            if f:
                vec_j1.append(x)
            else:
                vec_j0.append(x)
        print(t)
        return vec_f, vec_j1 if len(vec_j1) < len(vec_j0) else vec_j0

    def bin_generation(self):
        vec_x = list()
        for i in range(0, 2 ** self.number_variables):
            vec_x.append(self.int_to_byte(i))
        return vec_x

    def int_to_byte(self, x):
        vec = [0 for _ in range(self.number_variables)]
        i = self.number_variables - 1
        while x > 0:
            vec[i] = x % 2
            x = x // 2
            i = i - 1
        return vec


class ActivationFunction:
    '''
    Класс ФА, в зависимости от типа функции можем
    получить её значение или производную
    '''

    def __init__(self, type_fa: str):
        self.type = type_fa
        self.net = 0

    def get_value(self, net: float):
        self.net = net
        if self.type == '1':
            return 1 if net >= 0 else 0
        elif self.type == '2':
            f_net = 1 / (1 + np.exp(-net))
            return 1 if f_net >= 0.5 else 0

    def get_derivative(self):
        if self.type == '1':
            return 1
        elif self.type == '2':
            f_net = 1 / (1 + np.exp(-self.net))
            df_net = f_net * (1 - f_net)
            return df_net


def get_real_function(vec_weights, vec_sets, vec_centers, func: ActivationFunction):
    '''
    Функция для получения реальной функции по весам НС
    '''
    vec_y = list()
    for set_i in range(len(vec_sets)):
        vec_fi = [1]  # fi0 = 1
        net = vec_weights[0]
        for j in range(len(vec_centers)):
            tmp_sum = 0
            for i in range(len(vec_sets[set_i])):
                tmp_sum += (vec_sets[set_i][i] - vec_centers[j][i]) ** 2
            fi_x = np.exp(-tmp_sum)
            vec_fi.append(fi_x)
            net += vec_weights[j + 1] * fi_x
        real_output = func.get_value(net)
        vec_y.append(real_output)
    return vec_y


class RBFNeuron:
    '''
    Класс, RBF нерона. Выполняет расчёт НС и коррекцию весов
    '''
    def __init__(self, count_centers: int, vec_centers: list, func: ActivationFunction, bias=1.):
        self.weights = [0 for _ in range(count_centers + 1)]
        self.centers = vec_centers
        self.function = func
        self.bias = bias
        self.output = 0

    def calculate(self, inputs_set: list):
        vec_fi = [self.bias]
        net = self.weights[0]
        for j in range(len(self.centers)):
            tmp_sum = 0
            for i in range(len(inputs_set)):
                tmp_sum += (inputs_set[i] - self.centers[j][i]) ** 2
            fi_x = np.exp(-tmp_sum)
            vec_fi.append(fi_x)
            net += self.weights[j + 1] * fi_x
        real_output = self.function.get_value(net)
        return real_output, vec_fi

    def correct_weights(self, learning_rate: float, delta: float, vec_fi: list):
        for j in range(len(self.centers) + 1):
            self.weights[j] += learning_rate * delta * vec_fi[j] * self.function.get_derivative()

    def reset(self):
        self.weights = [0 for _ in range(len(self.weights))]


def learning(vec_func, vec_centers, vec_sets, func: ActivationFunction, learning_rate):
    '''
    Функция обучения НС
    '''
    vec_error = list()
    neuron = RBFNeuron(len(vec_centers), vec_centers, func)
    for number_sets in range(2, len(vec_sets)):
        for setX, setF in zip(itertools.combinations(vec_sets, number_sets),
                              itertools.combinations(vec_func, number_sets)):
            vec_y = get_real_function(neuron.weights, setX, vec_centers, func)
            error = hamming_distance(setF, vec_y)
            table = PrettyTable(['K', 'W', 'Y', 'E'])
            table.add_row([len(vec_error), copy.copy(neuron.weights), copy.copy(vec_y), error])
            vec_error.append(error)
            while error != 0 and len(vec_error) < 100:
                for set_i in range(len(setX)):
                    real_output, vec_fi = neuron.calculate(setX[set_i])
                    delta = setF[set_i] - real_output
                    neuron.correct_weights(learning_rate, delta, vec_fi)
                vec_y = get_real_function(neuron.weights, setX, vec_centers, func)
                error = hamming_distance(setF, vec_y)
                table.add_row([len(vec_error), copy.copy(neuron.weights), copy.copy(vec_y), error])
                vec_error.append(error)
            vec_y = get_real_function(neuron.weights, vec_sets, vec_centers, func)
            if hamming_distance(vec_func, vec_y) == 0:
                print('Удалось обучить на', number_sets, 'наборах')
                for i in range(number_sets):
                    print('X' + str(i + 1), '=', setX[i], end=' ')
                print()
                print(table)
                return neuron.weights, vec_error
            else:
                neuron.reset()
                vec_error.clear()
                table.clear()


if __name__ == "__main__":
    number_variables = 4
    learning_rate = float(input('Введите норму обучения:'))
    activation_func = ActivationFunction(input('Введите ФА ("1" - пороговая ФА, "2" - сигмоидальная ФА):'))

    tt = TruthTable(number_variables)
    target_function, centers_rbf_neurons = tt.init_parameters()
    print('Центры RBF:', centers_rbf_neurons)

    sets_x = tt.bin_generation()

    weights, vec_errors = learning(target_function, centers_rbf_neurons, sets_x, activation_func, learning_rate)

    real_function = get_real_function(weights, sets_x, centers_rbf_neurons, activation_func)
    print('Целевая функция', target_function)
    print('Реальная функция', real_function)

    line_plot([i for i in range(len(vec_errors))], vec_errors, "Error E", "Era K", "E(k)")
