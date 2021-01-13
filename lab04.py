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


class ActivationFunction:
    '''
    Класс ФА, в зависимости от типа функции можем
    получить её значение или производную
    '''

    def __init__(self):
        self.net = 0

    def get_value(self, net: float):
        return (1 - np.exp(-net)) / (1 + np.exp(-net))

    def get_derivative(self, net: float):
        f_net = (1 - np.exp(-net)) / (1 + np.exp(-net))
        return (1 - f_net ** 2) / 2


class Neuron:
    '''
    Класс, RBF нерона. Выполняет расчёт НС и коррекцию весов
    '''

    def __init__(self, number_weights: int, func: ActivationFunction, bias=1):
        self.weights = [0. for _ in range(number_weights + 1)]
        self.function = func
        self.output = 0
        self.net = 0
        self.bias = bias

    def calculate(self, inputs: list):
        self.net = self.weights[0]
        for i in range(1, len(self.weights)):
            self.net += self.weights[i] * inputs[i]
        self.output = self.function.get_value(self.net)

    def correct_weights(self, learning_rate: float, inputs: list, delta: float):
        self.weights[0] += learning_rate * self.bias * delta
        for i in range(1, len(self.weights)):
            self.weights[i] += learning_rate * inputs[i] * delta


def get_error(vec_t, vec_y):
    delta_sum = 0
    for j in range(len(vec_t)):
        delta_sum += (vec_t[j] - vec_y[j]) ** 2
    return np.sqrt(delta_sum)


def get_y(hidden_layer_neurons, output_layer_neurons, vec_x):
    vec_y = list()
    vec_out = [1]
    for j in range(len(hidden_layer_neurons)):
        hidden_layer_neurons[j].calculate(vec_x)
        vec_out.append(hidden_layer_neurons[j].output)
    for m in range(len(output_layer_neurons)):
        output_layer_neurons[m].calculate(vec_out)
        vec_y.append(output_layer_neurons[m].output)
    return vec_y


def calculate_layer(neurons: list, inputs: list, hidden: bool):
    vec_out = [neurons[0].bias] if hidden else list()
    vec_net = list()
    for j in range(len(neurons)):
        neurons[j].calculate(inputs)
        vec_out.append(neurons[j].output)
        vec_net.append(neurons[j].net)
    return vec_out, vec_net


def calculate_delta_m(architecture, vec_t, vec_net, vec_out, func):
    delta_m = list()
    for m in range(architecture[2]):
        delta = func.get_derivative(vec_net[m]) * (vec_t[m] - vec_out[m])
        delta_m.append(delta)
    return delta_m


def calculate_delta_j(architecture, neurons, delta_m, vec_net, func):
    delta_j = list()
    for j in range(architecture[1]):
        tmp_sum = 0
        for m in range(architecture[2]):
            tmp_sum += neurons[m].weights[j + 1] * delta_m[m]
        delta = func.get_derivative(vec_net[j]) * tmp_sum
        delta_j.append(delta)
    return delta_j


def learning(architecture: list, vec_x: list, vec_t: list, learning_rate: float, func: ActivationFunction, epsilon):
    table = PrettyTable(['Era', 'Hidden layer weights', 'Output layer weights', 'Y', 'Error'])
    # Инициализация слоёв
    hidden_layer_neurons = [Neuron(architecture[0], func) for _ in range(architecture[1])]
    output_layer_neurons = [Neuron(architecture[1], func) for _ in range(architecture[2])]

    hidden_weights = [hidden_layer_neurons[i].weights for i in range(len(hidden_layer_neurons))]
    output_weights = [output_layer_neurons[i].weights for i in range(len(output_layer_neurons))]
    vec_y = get_y(hidden_layer_neurons, output_layer_neurons, vec_x)
    error = get_error(vec_t, vec_y)
    vec_error = list()
    # Формирование первой строки таблицы
    table.add_row([len(vec_error),
                   [[float('{:.3}'.format(w)) for w in hidden_weights[i]] for i in range(len(hidden_weights))],
                   [[float('{:.3}'.format(w)) for w in output_weights[j]] for j in range(len(output_weights))],
                   [float('{:.3}'.format(y)) for y in vec_y], round(error, 6)])
    vec_error.append(error)
    while error > epsilon:
        # Этап 1
        vec_out1, vec_net1 = calculate_layer(hidden_layer_neurons, vec_x, True)
        vec_out2, vec_net2 = calculate_layer(output_layer_neurons, vec_out1, False)

        # Этап 2
        delta_m = calculate_delta_m(architecture, vec_t, vec_net2, vec_out2, func)
        delta_j = calculate_delta_j(architecture, output_layer_neurons, delta_m, vec_net1, func)

        # Этап 3
        [hidden_layer_neurons[j].correct_weights(learning_rate, vec_x, delta_j[j]) for j in
         range(len(hidden_layer_neurons))]
        [output_layer_neurons[m].correct_weights(learning_rate, vec_out1, delta_m[m]) for m in
         range(len(output_layer_neurons))]

        # Расчёт выхода по полученым весам
        vec_y = get_y(hidden_layer_neurons, output_layer_neurons, vec_x)
        error = get_error(vec_t, vec_y)

        # Формирование таблицы
        hidden_weights = [hidden_layer_neurons[i].weights for i in range(len(hidden_layer_neurons))]
        output_weights = [output_layer_neurons[i].weights for i in range(len(output_layer_neurons))]
        table.add_row([len(vec_error),
                       [[float('{:.3}'.format(w)) for w in hidden_weights[i]] for i in range(len(hidden_weights))],
                       [[float('{:.3}'.format(w)) for w in output_weights[j]] for j in range(len(output_weights))],
                       [float('{:.3}'.format(y)) for y in vec_y], round(error, 6)])
        vec_error.append(error)
    print(table)
    return vec_y, hidden_weights, output_weights, vec_error, table


if __name__ == "__main__":
    NMJ = [1, 1, 3]
    x = [1, -1]
    t = [2, -3, 1]
    norm = 1
    epsilon = 0.001
    fa = ActivationFunction()
    real_y, w1, w2, errors, table = learning(NMJ, x, [t[i] / 10 for i in range(len(t))], norm, fa, epsilon)
    # line_plot([i for i in range(len(errors))], errors, "Error E", "Era K", "E(k)")
