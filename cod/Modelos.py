import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

class PreparacionDatos:
    def __init__(self, df, target_variable):
        self.df = df
        self.target_variable = target_variable
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def extraer_variables(self):
        self.X = self.df.drop([self.target_variable], axis=1)
        self.Y = self.df[self.target_variable]
        print(f"Variables independientes (X) e independiente (Y) extraídas. Variable objetivo: '{self.target_variable}'")

    def dividir_datos(self, train_size=0.75, random_state=1234, shuffle=True):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, train_size=train_size, random_state=random_state, shuffle=shuffle
        )
        print("Datos divididos en conjuntos de entrenamiento y prueba.")
        print(f"Tamaño del conjunto de entrenamiento: {len(self.X_train)}")
        print(f"Tamaño del conjunto de prueba: {len(self.X_test)}")

class Graficos:
    @staticmethod
    def plot_confusion_matrix(Y_test, predictions):
        matrix = confusion_matrix(Y_test, predictions)
        plot_confusion_matrix(conf_mat=matrix, figsize=(6,6), show_normed=False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def feature_importance(importances, variables, title):
        permutation_importancias = pd.DataFrame({"Feature": variables, "Importance": importances})
        print(permutation_importancias)

        plt.figure(figsize=(10, 6))
        plt.barh(permutation_importancias["Feature"], permutation_importancias["Importance"], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.tight_layout()
        plt.show()

class LinearSVM(PreparacionDatos):
    def __init__(self, df, target_variable, C=100, random_state=123):
        PreparacionDatos.__init__(self, df, target_variable)
        self.model = SVC(C=C, kernel='linear', random_state=random_state)
        self.extraer_variables()
        self.dividir_datos()

    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self):
        return self.model.predict(self.X_test)

    def accuracy(self):
        predictions = self.predict()
        accuracy = accuracy_score(y_true=self.Y_test, y_pred=predictions, normalize=True)
        print(f"La precisión del test es: {100*accuracy}%")
        return accuracy

    def plot_confusion_matrix(self):
        predictions = self.predict()
        Graficos.plot_confusion_matrix(self.Y_test, predictions)

    def feature_importance(self, n_repeats=10, random_state=123, n_jobs=2):
        result = permutation_importance(self.model, self.X, self.Y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        importances = result.importances_mean
        variables = self.X.columns
        Graficos.feature_importance(importances, variables, 'Importancias caso Lineal')

class RadialSVM(PreparacionDatos):
    def __init__(self, df, target_variable, param_grid=None, cv=3, scoring='accuracy', n_jobs=-1, verbose=0):
        PreparacionDatos.__init__(self, df, target_variable)
        if param_grid is None:
            param_grid = {'C': np.logspace(-5, 7, 20)}
        self.grid_search = GridSearchCV(
            estimator=SVC(kernel='rbf', gamma='scale'),
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
            return_train_score=True
        )
        self.best_model = None
        self.extraer_variables()
        self.dividir_datos()

    def fit(self):
        self.grid_search.fit(X=self.X_train, y=self.Y_train)
        self.best_model = self.grid_search.best_estimator_

        print("----------------------------------------")
        print("Mejores hiperparámetros encontrados (cv)")
        print("----------------------------------------")
        print(self.grid_search.best_params_, ":", self.grid_search.best_score_, self.grid_search.scoring)

        return pd.DataFrame(self.grid_search.cv_results_)\
            .filter(regex='(param.*|mean_t|std_t)')\
            .drop(columns='params')\
            .sort_values('mean_test_score', ascending=False)\
            .head(5)

    def predict(self):
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        return self.best_model.predict(self.X_test)

    def accuracy(self):
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        predictions = self.predict()
        accuracy = accuracy_score(y_true=self.Y_test, y_pred=predictions, normalize=True)
        print(f"La precisión del test es: {100*accuracy}%")
        return accuracy
    
    def plot_confusion_matrix(self):
        predictions = self.predict()
        Graficos.plot_confusion_matrix(self.Y_test, predictions)

    def feature_importance(self, n_repeats=10, random_state=123, n_jobs=2):
        result = permutation_importance(self.best_model, self.X, self.Y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        importances = result.importances_mean
        variables = self.X.columns
        Graficos.feature_importance(importances, variables, 'Importancias caso Radial')
