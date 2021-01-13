import numpy as np


def print_image(size_i, size_j, vec_image):
    '''
       Функция печати биполярных кодов образов
    '''
    for i in range(size_i):
        for n in range(len(vec_image)):
            assert len(vec_image[n]) == size_i * size_j
            for j in range(size_j):
                if vec_image[n][i + size_i * j] == 1:
                    print(f'{1:3}', end='')
                else:
                    print(f'{"  -":3}', end='')
            print(f'{"":3}', end='')
        print()


class ActivationFunction:
    '''
    Класс функции активации
    '''

    def __init__(self):
        self.net = 0

    def get_value(self, net: int, f_net: int):
        if net > 0:
            return 1
        elif net < 0:
            return -1
        else:
            return f_net


class Hopfild_Network:
    '''
        Класс РНС Хопфилда, выполняет рассчёт матрицы
        весов, и расспознавание искаженных образов
    '''

    def __init__(self, size: int, func: ActivationFunction):
        self.size = size
        self.func = func
        self.weight_matrix = np.zeros((size, size), dtype=int)

    def calculate_matrix_weights(self, vec_standards):
        for j in range(self.size):
            for k in range(self.size):
                if j == k:
                    self.weight_matrix[j][k] = 0
                else:
                    self.weight_matrix[j][k] = sum(
                        [vec_standards[l][j] * vec_standards[l][k] for l in range(len(vec_standards))])

    def recognition(self, vec_input):
        vec_y = [0 for _ in range(len(vec_input))]
        while not (np.array_equal(vec_y, vec_input)):
            for k in range(len(vec_input)):
                net = 0;
                for j in range(len(vec_input)):
                    net += self.weight_matrix[j][k] * vec_input[j]
                    print(f'{self.weight_matrix[j][k]} * {vec_input[j]} = {self.weight_matrix[j][k] * vec_input[j]}')
                vec_y[k] = self.func.get_value(net, vec_input[k])
                print(f'net = {net}, y = {vec_y[k]}')
            vec_input = vec_y
        return vec_y


if __name__ == "__main__":
    # size_I = 7
    # size_J = 5
    # vec_A = [-1, -1, 1, 1, 1, 1, 1,
    #          -1, 1, -1, 1, -1, -1, -1,
    #          1, -1, -1, 1, -1, -1, -1,
    #          -1, 1, -1, 1, -1, -1, -1,
    #          -1, -1, 1, 1, 1, 1, 1]
    # vec_I = [1, -1, -1, -1, -1, -1, 1,
    #          1, -1, -1, -1, -1, -1, 1,
    #          1, 1, 1, 1, 1, 1, 1,
    #          1, -1, -1, -1, -1, -1, 1,
    #          1, -1, -1, -1, -1, -1, 1]
    # vec_F = [1, 1, 1, 1, 1, 1, 1,
    #          1, -1, -1, 1, -1, -1, -1,
    #          1, -1, -1, 1, -1, -1, -1,
    #          1, -1, -1, -1, -1, -1, -1,
    #          1, -1, -1, -1, -1, -1, -1]
    size_I = 2
    size_J = 2
    vec_A = [1, 1, 1, -1]
    vec_I = [-1, -1, 1, 1]


    vector_standards = [vec_A, vec_I]
    print(f"Матрицы-паттерны размерности {size_I}x{size_J}: A, I, F ('-1' = '-' для наглядности)")
    print_image(size_I, size_J, vector_standards)

    func = ActivationFunction()
    network = Hopfild_Network(size_I * size_J, func)
    network.calculate_matrix_weights(vector_standards)

    print('Матрица весосв:')
    print(network.weight_matrix)

    print('Подадим для тестирования поочерёдно все три рабочих вектора на вход. Результат работы сети:')
    # y_A = network.recognition(vec_A)
    # y_I = network.recognition(vec_I)
    # print_image(size_I, size_J, [y_A, y_I])

    print('Исказим входной образ A, инвертируем 3, 8, 17, 25, 34 биты. Результат работы сети:')
    distorted_vec_A = [1,-1,-1,1]
    distorted_y_A = network.recognition(distorted_vec_A)
    print_image(size_I, size_J, [distorted_vec_A, distorted_y_A])
    #
    # print('Исказим входной образ I, инвертируем 1, 8, 17, 20, 32 биты. Результат работы сети:')
    # distorted_vec_I = [-1, -1, -1, -1, -1, -1, 1,
    #                    -1, -1, -1, -1, -1, -1, 1,
    #                    1, 1, -1, 1, 1, -1, 1,
    #                    1, -1, -1, -1, -1, -1, 1,
    #                    1, -1, -1, 1, -1, -1, 1]
    # distorted_y_I = network.recognition(distorted_vec_I)
    # print_image(size_I, size_J, [distorted_vec_I, distorted_y_I])
    #
    # print('Исказим входной образ F, инвертируем 2, 5, 18, 24, 30 биты. Результат работы сети:')
    # distorted_vec_F = [1, -1, 1, 1, -1, 1, 1,
    #                    1, -1, -1, 1, -1, 1, -1,
    #                    1, -1, -1, -1, -1, -1, -1,
    #                    1, -1, 1, -1, -1, -1, -1,
    #                    1, 1, -1, -1, -1, -1, -1]
    # distorted_y_F = network.recognition(distorted_vec_F)
    # print_image(size_I, size_J, [distorted_vec_F, distorted_y_F])
