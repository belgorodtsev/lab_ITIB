import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Данные колледжей
# https://data.mos.ru/opendata/546

# Цвета для центров - округов москвы
COLORS = ['y', 'b', 'r', 'g', 'c', 'm', 'lime', 'gold', 'orange', 'coral', 'purple', 'grey']

DISTRICT = {"Восточный административный округ": [55.787710, 37.775631],
            "Западный административный округ": [55.728003, 37.443533],
            "Зеленоградский административный округ": [55.987583, 37.194250],
            "Новомосковский административный округ": [55.558121, 37.370724],
            "Северный административный округ": [55.838384, 37.525765],
            "Северо-Восточный административный округ": [55.863894, 37.620923],
            "Северо-Западный административный округ": [55.829370, 37.451546],
            "Троицкий административный округ": [55.355771, 37.146990],
            "Центральный административный округ": [55.753995, 37.614069],
            "Юго-Восточный административный округ": [55.692019, 37.754583],
            "Юго-Западный административный округ": [55.662735, 37.576178],
            "Южный административный округ": [55.610906, 37.681479]}

NAME_DISTRICT = ['ВАО', 'ЗАО', 'ЗелАО', 'Новомосковский АО', 'САО', 'СВАО', 'СЗАО', 'Троицкий АО', 'ЦАО', 'ЮВАО',
                 'ЮЗАО', 'ЮАО']


def get_data(url, filename):
    """
    Функция делает POST запрос и создаёт файл с полученными данными в формате .json
    """
    URL = url
    client = requests.session()
    client.get(URL)
    res = requests.post(URL, headers=dict(Referer=URL))

    with open(filename, 'w') as outfile:
        json.dump(res.json(), outfile, ensure_ascii=False, separators=(',', ': '), indent=4, sort_keys=False)


class KohonenNetwork:
    """
    НС Кохонена
    """

    def __init__(self, values, centers):
        """
        Конструктор класса
        :param values: массив из (x, y) значений для кластеризации
        :param centers: массив из (x, y) центров кластеров
        """
        self.values = np.array(values)
        self.centers = np.array(centers)

        # Матрица весов
        self.weights = np.zeros((len(values), len(centers)))

    def euclidean_distance(self, a, b):
        """
        Функция расчета Евклидова расстояния между матрицами а и b
        """
        return np.linalg.norm(a - b)

    def calculate_weights(self):
        """
        Вычисление матрицы весов - принадлежности к кластерам
        """

        # Считаем расстояние для каждого нейрона и входного сигнала
        for value_i in range(len(self.values)):
            for center_i in range(len(self.centers)):
                self.weights[value_i][center_i] = self.euclidean_distance(self.values[value_i], self.centers[center_i])

        # Выполнение правила сильнейшего: минимальному элементу присваиваем 1, остальным 0
        for value_i in range(len(self.values)):
            min_index = self.weights[value_i].argmin()
            self.weights[value_i][min_index] = 1
            self.weights[value_i][0:min_index] = 0
            self.weights[value_i][min_index + 1:] = 0

        return self.weights


class ClusterAnalysis():
    def __init__(self, data_colleges):
        self.read_json(data_colleges)
        _, self.ax = plt.subplots()
        self.save_data('init.png')

    def read_json(self, data_colleges):
        """
        Функция считывания датасета из .json файлов
        """
        # Парсинг координат колледжей
        json_data = open(data_colleges).read()
        data = json.loads(json_data)
        colleges_data = [data['features'][i]['geometry']['coordinates'] for i in
                         range(len(data['features']))]
        dist_data = [data['features'][i]['properties']['Attributes']['okrug'] for i in
                     range(len(data['features']))]

        name_data = [data['features'][i]['properties']['Attributes']['name'] for i in
                     range(len(data['features']))]

        # для более удобной работы с данными создадим DataFrame
        colleges = pd.DataFrame(colleges_data, columns=['x', 'y'])
        colleges['districts'] = dist_data
        colleges['color'] = 'k'  # изначально цвет колледжей чёрный
        colleges['size'] = 6

        colleges['name'] = name_data

        self.colleges = colleges

        # Парсинг координат центров округов
        districts_data = DISTRICT.values()
        districts = pd.DataFrame(districts_data, columns=['y', 'x'])
        districts['color'] = COLORS  # каджому округу Москвы соответствует свой цвет
        districts['size'] = 26
        self.districts = districts

    def save_data(self, filename):
        self.ax.scatter(x=self.colleges['x'], y=self.colleges['y'],
                        s=self.colleges['size'],
                        c=self.colleges['color'])

        for i in range(len(COLORS)):
            self.ax.scatter(x=self.districts['x'][i],
                            y=self.districts['y'][i],
                            s=self.districts['size'][i],
                            marker='s',
                            c=COLORS[i],
                            label=NAME_DISTRICT[i])

        self.ax.legend(fontsize=7, loc='lower right')
        self.ax.axis('off')
        plt.savefig(filename)

    def clustering(self):
        """
        Функция, выполняющая кластерный анализ.
        Перекрашивает точки соответствующими цветами в соответствии с новыми кластерами
        """
        network = KohonenNetwork(self.colleges[['x', 'y']].values, self.districts[['x', 'y']].values)
        weights = network.calculate_weights()

        colleges_data = self.colleges.values  # таблица, где столбец 0-x, 1-y, 2-districts, 3-color, 4-size
        districts_data = self.districts.values

        error_list = list()
        # перекрашиваем точки
        for i in range(len(colleges_data)):
            # получаем индекс элемента в строке где стоит 1 и красим в цвет соответствующи округу
            center = weights[i].argmax()
            colleges_data[i][3] = COLORS[center]
            if colleges_data[i][2] != list(DISTRICT.keys())[center]:
                # Если нам необходимо не закрашивать точки распределённые направильно
                # colleges_data[i][3] = 'k'
                error_list.append(
                    f'{colleges_data[i][5]} должен принадлежать {colleges_data[i][2]}, а пренадлежит {list(DISTRICT.keys())[center]}')

        print(len(error_list))
        self.colleges['color'] = colleges_data.T[3]
        self.districts['color'] = districts_data.T[3]

        self.ax.clear()
        self.save_data('result.png')
        plt.show()


if __name__ == "__main__":
    # Получение данных для анализа
    # get_data('https://apidata.mos.ru/v1/datasets/546/features', 'colleges.json')

    # В результате выполнения сохранится картинка с исходными данными
    analysis = ClusterAnalysis('colleges.json')

    # В результате выполнения сохранится картинка кластеризованными данными
    analysis.clustering()
