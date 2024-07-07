import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


class PrecisionImportancias:
    def __init__(self, data_path):
        self.__data_path = data_path
        self.__data = pd.read_csv(data_path)
        self.__x_data = self.__data.drop(["Outcome"], axis=1)
        self.__y = self.__data.Outcome.values
        self.__x = (self.__x_data - np.min(self.__x_data)) / (np.max(self.__x_data) - np.min(self.__x_data))
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__x, self.__y, test_size=0.2, random_state=73)
        self.__dt = DecisionTreeClassifier()
        self.__feature_importance_df = None
        self.__accuracy = None
        self.__train_model()

    def __train_model(self):
        self.__dt.fit(self.__x_train, self.__y_train)
        self.__accuracy = self.__dt.score(self.__x_test, self.__y_test)
        
        feature_importances = self.__dt.feature_importances_
        self.__feature_importance_df = pd.DataFrame({
            'Feature': self.__x.columns,
            'Importance': feature_importances
        })

    def display_results(self):
        print(f"Puntuación de precisión: {self.__accuracy:.2f}")
        for feature, importance in zip(self.__feature_importance_df['Feature'], self.__feature_importance_df['Importance']):
            print(f"{feature}: {importance:.2f}")

    @property
    def data_path(self):
        return self.__data_path

    @data_path.setter
    def data_path(self, value):
        self.__data_path = value
        self.__data = pd.read_csv(value)
        self.__x_data = self.__data.drop(["Outcome"], axis=1)
        self.__y = self.__data.Outcome.values
        self.__x = (self.__x_data - np.min(self.__x_data)) / (np.max(self.__x_data) - np.min(self.__x_data))
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__x, self.__y, test_size=0.2, random_state=73)
        self.__dt = DecisionTreeClassifier()
        self.__train_model()

    @property
    def accuracy(self):
        return self.__accuracy
    
    @property
    def feature_importance_df(self):
        return self.__feature_importance_df

    @property
    def dt(self):
        return self.__dt

    @property
    def data(self):
        return self.__data

    def __str__(self):
        return f'PrecisionImportancias(data_path={self.__data_path}, accuracy={self.__accuracy})'



class PrecisionImportanciasGrafico:
    def __init__(self, feature_importance_df):
        self.__feature_importance_df = feature_importance_df
        self.__plot_feature_importances()
    
    @property
    def feature_importance_df(self):
        return self.__feature_importance_df

    def __plot_feature_importances(self):
        plt.figure(figsize=(10, 6))
        plt.barh(self.__feature_importance_df['Feature'], self.__feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.show()

    def __str__(self):
        return 'PrecisionImportanciasGrafico()'


class ArbolDecision:
    def __init__(self, model, data_columns):
        self.__model = model
        self.__data_columns = data_columns
        self.__plot_tree()
    
    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, value):
        self.__model = value
    
    @property
    def data_columns(self):
        return self.__data_columns
    
    @data_columns.setter
    def data_columns(self, value):
        self.__data_columns = value
    
    def __plot_tree(self):
        plt.figure(figsize=(400, 200))
        plot_tree(self.__model, feature_names=self.__data_columns, class_names=['Non-Diabetic', 'Diabetic'], filled=True)
        plt.show()

    def __str__(self):
        return f'ArbolDecision(model={self.__model}, data_columns={self.__data_columns})'









class PromediosPrecisionImportancias:
    def __init__(self, data_path, n_iterations):
        self.data_path = data_path
        self.n_iterations = n_iterations
        self.mean_accuracy = None
        self.mean_feature_importance_df = None
        
        np.random.seed(3435)  # set seed para tener consistencia al ejecutar sobre una misma cantidad
        self._run_iterations()
        self._print_results()
        self.plot_feature_importances()

    def _run_iterations(self):
        accuracies = []
        feature_importances_list = []

        for _ in range(self.n_iterations):
            prec_import = PrecisionImportancias(self.data_path)
            accuracies.append(prec_import.accuracy)
            feature_importances_list.append(prec_import.feature_importance_df['Importance'].values)

        self.mean_accuracy = np.mean(accuracies)
        mean_feature_importances = np.mean(feature_importances_list, axis=0)
        self.mean_feature_importance_df = pd.DataFrame({
            'Feature': prec_import.feature_importance_df['Feature'],
            'Importance': mean_feature_importances
        })

    def _print_results(self):
        print(f"Puntuación de precisión promedio: {self.mean_accuracy:.2f}")
        for feature, importance in zip(self.mean_feature_importance_df['Feature'], self.mean_feature_importance_df['Importance']):
            print(f"{feature}: {importance:.2f}")

    def plot_feature_importances(self):
        plt.figure(figsize=(10, 6))
        plt.barh(self.mean_feature_importance_df['Feature'], self.mean_feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Mean Feature Importances (Iterations: {self.n_iterations})')  # Incluimos el número de iteraciones en el título
        plt.tight_layout()
        plt.show()



# data = "D:\\Descargas\\diabetes.csv"

# prec_import = PrecisionImportancias(data)

# prec_import_grafico = PrecisionImportanciasGrafico(prec_import.feature_importance_df)

# arbol_decision = ArbolDecision(prec_import.dt, prec_import.data.columns[:-1])


# n_iterations = 100
# iter_prec_import = PromediosPrecisionImportancias(data, n_iterations)

