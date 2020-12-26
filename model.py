import pandas
import sklearn
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style


class Model:
    def __init__(self):
        self.data = pandas.read_csv('student-mat.csv', sep=';')
        self.convert_to_int()
        self.split_data()
        self.fit()
        self.plot()

    def convert_to_int(self):
        converter = preprocessing.LabelEncoder()
        self.sex = converter.fit_transform(list(self.data['sex']))
        self.st = converter.fit_transform(list(self.data['studytime']))
        self.absences = converter.fit_transform(list(self.data['absences']))
        self.g1 = converter.fit_transform(list(self.data['G1']))
        self.g2 = converter.fit_transform(list(self.data['G2']))
        self.g3 = converter.fit_transform(list(self.data['G3']))

    def split_data(self):
        x = list(zip(self.sex, self.st, self.absences, self.g1, self.g2, self.g3))
        y = list(self.g3)
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(x, y,
                                                                                                        test_size=0.1)

    def fit(self):
        self.model = linear_model.LinearRegression()
        self.model.fit(self.x_train, self.y_train)

    def plot(self):
        x = 'absences'
        y = 'G3'
        style.use('ggplot')
        pyplot.title('Student Performance')
        pyplot.scatter(self.data[x], self.data[y])
        pyplot.xlabel('Absences')
        pyplot.ylabel('Final Grade')
        pyplot.show()


model = Model()
