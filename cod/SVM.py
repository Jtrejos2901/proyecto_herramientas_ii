from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, C=100, random_state=123):
        self.model = SVC(C=C, kernel='linear', random_state=random_state)

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def accuracy(self, X_test, Y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_true=Y_test, y_pred=predictions, normalize=True)
        print(f"La precisión del test es: {100*accuracy}%")
        return accuracy

    def plot_confusion_matrix(self, X_test, Y_test):
        predictions = self.predict(X_test)
        matrix = confusion_matrix(Y_test, predictions)
        plot_confusion_matrix(conf_mat=matrix, figsize=(6,6), show_normed=False)
        plt.tight_layout()
        plt.show()

    def feature_importance(self, X, Y, n_repeats=10, random_state=123, n_jobs=2):
        result = permutation_importance(self.model, X, Y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        importances = result.importances_mean
        variables = X.columns
        permutation_importancias_1 = pd.DataFrame({"Feature": variables, "Importance": importances})
        print(permutation_importancias_1)

        plt.figure(figsize=(10, 6))
        plt.barh(permutation_importancias_1["Feature"], permutation_importancias_1["Importance"], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Importancias caso Lineal')
        plt.tight_layout()
        plt.show()


class RadialSVM:
    def __init__(self, param_grid=None, cv=3, scoring='accuracy', n_jobs=-1, verbose=0):
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

    def fit(self, X_train, Y_train):
        self.grid_search.fit(X=X_train, y=Y_train)
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

    def predict(self, X_test):
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        return self.best_model.predict(X_test)

    def accuracy(self, X_test, Y_test):
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_true=Y_test, y_pred=predictions, normalize=True)
        print(f"La precisión del test es: {100*accuracy}%")
        return accuracy
    
    def plot_confusion_matrix(self, X_test, Y_test):
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        predictions = self.predict(X_test)
        matrix = confusion_matrix(Y_test, predictions)
        plot_confusion_matrix(conf_mat=matrix, figsize=(6,6), show_normed=False)
        plt.tight_layout()
        plt.show()

    def feature_importance(self, X, Y, n_repeats=10, random_state=123, n_jobs=2):
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        result = permutation_importance(self.best_model, X, Y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        importances = result.importances_mean
        variables = X.columns
        permutation_importancias = pd.DataFrame({"Feature": variables, "Importance": importances})
        print(permutation_importancias)

        plt.figure(figsize=(10, 6))
        plt.barh(permutation_importancias["Feature"], permutation_importancias["Importance"], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Importancias Caso Radial')
        plt.tight_layout()
        plt.show()