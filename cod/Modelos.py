import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier # Para el RF
from sklearn.tree import DecisionTreeClassifier # Para el ADD
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Se crea la clase, PreparacionDatos la cual se encargará de la división de los 
# en los conjuntos de entranamiento y prueba. 
class PreparacionDatos:
    # Se crea el constructor de la clase.
    def __init__(self, df, target_variable):
        '''
        Método contructor de la clase PreparacionDatos.

        Parameters:
            df (pandas.DataFrame): El dataframe o tabla con los datos a
                                   particionar.
            target_variable (string): EL nombre de la variable a predecir.
            
        Returns:
            None.
        '''
        self.df = df
        self.target_variable = target_variable
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
    
    # Mñetodo para la extracción de variables.
    def extraer_variables(self):
        '''
        Método para extraer y definir las variables prdictorias y la variable a
        predecir.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        # Se definenen los atributos x e y de la clase.
        self.X = self.df.drop([self.target_variable], axis=1)
        self.Y = self.df[self.target_variable]
        
        # Se imprimen los resultados.
        print(f"Variables independientes (X) e independiente (Y) extraídas. Variable objetivo: '{self.target_variable}'")

    def dividir_datos(self, train_size=0.75, random_state=1234, shuffle=True):
        '''
        Método para la división de los datos en dos conjuntos, uno para pruebas
        y otro para entrenamiento.

        Parameters:
            train_size (float): Porcentaje de la base de datos que se destinará
                                al conjunto de entrenamiento.
            random_state (int): Valor equivalente a la semilla para la 
                                reproducibilidad de la división del conjunto de
                                datos inicial.
            shuffle (bool): Parámetro que determina si se mezclan los datos 
                            antes de la división.
            
        Returns:
            None.
        '''
        # Se divide la base de datos.
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, train_size=train_size, random_state=random_state, shuffle=shuffle
        )
        
        # Se imprime información relevante.
        print("Datos divididos en conjuntos de entrenamiento y prueba.")
        print(f"Tamaño del conjunto de entrenamiento: {len(self.X_train)}")
        print(f"Tamaño del conjunto de prueba: {len(self.X_test)}")

# Se crea la clase para la generación de gráficos.
class Graficos:
    # Método para generar el gráfico de la matriz de confunción.
    @staticmethod
    def plot_confusion_matrix(Y_test, predictions):
        '''
        Método destinado a la generación de los gráficos de la matriz de 
        confunción.

        Parameters:
            Y_test (list): Lista que contiene los valores de la variable
                           a predecir para el conjunto de prueba.
            predictions (list): Lista con los valores de la variable a predecir
                                obtenidos por el modelo.

        Returns:
            None.
        '''
        # Se genera la matriz de confunción.
        matrix = confusion_matrix(Y_test, predictions)
        
        # Se crea el gráfico
        plot_confusion_matrix(conf_mat=matrix, figsize=(6,6), show_normed=False)
        plt.tight_layout()
        
        # Se muetra el gráfico reciñen generado.
        plt.show()

    @staticmethod
    def feature_importance(importances, variables, title):
        '''
        Método destinado a la generación de los gráficos de las importancias
        de cada variable para el modelo.

        Parameters:
            importances (list): Lista con los valores de importancia.
            variables (list): Lista con el nombre de las variables.
            title (String): Titulo que llevará el gráfico.

        Returns:
            None.
        '''
        # Se genera un dataframe para graficar.
        permutation_importancias = pd.DataFrame({"Feature": variables, "Importance": importances})
        print(permutation_importancias)
        
        # Se genera el gráfico.
        plt.figure(figsize=(10, 6))
        plt.barh(permutation_importancias["Feature"], permutation_importancias["Importance"], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.tight_layout()
        
        # Se uestra el gráfico.
        plt.show()

# Se crea la clase para el modelo SVM con Kernel lineal.
class LinearSVM(PreparacionDatos):
    # Se crea el constructor de la clase.
    def __init__(self, df, target_variable, C=100, random_state=123):
        '''
        Método contructor de la clase LinearSVM.

        Parameters:
            df (pandas.DataFrame): El dataframe o tabla con los datos a
                                   particionar.
            target_variable (string): EL nombre de la variable a predecir.
            C (float): Parámetro de regularización. 
            random_state (int): Semiila para la reproducibilidad de los 
                                resultados.
            
        Returns:
            None.
        '''
        PreparacionDatos.__init__(self, df, target_variable)
        self.model = SVC(C=C, kernel='linear', random_state=random_state)
        self.extraer_variables()
        self.dividir_datos()
    
    # Método para ajustar el modelo.
    def fit(self):
        '''
        Método para ajustar el modelo de Suport Vector Machine con Kernel 
        lineal.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        self.model.fit(self.X_train, self.Y_train)
        
    # Método para generar las prediciones del modelo.
    def predict(self):
        '''
        Método para generar la prediciones del modelo de Suport Vector Machine 
        con Kernel lineal.

        Parameters:
            None.
            
        Returns:
            self.model.predict (ndarray): Arreglo con las predicicones.
        '''
        return self.model.predict(self.X_test)
    
    # Método para obtener la precisión del modelo.
    def accuracy(self):
        '''
        Método para obtener la precisión del modelo de Suport Vector Machine 
        con Kernel lineal.

        Parameters:
            None.
            
        Returns:
            accuracy (float): Nivel de precisión del modelo.
        '''
        # Se obtienen las predicciones.
        predictions = self.predict()
        
        # Se obtiene el nivel de precisión del modelo.
        accuracy = accuracy_score(y_true=self.Y_test, y_pred=predictions, normalize=True)
        print(f"La precisión del test es: {100*accuracy}%")
        return accuracy
    
    # Método para generar el gráfico de la matriz de confunción.
    def plot_confusion_matrix(self):
        '''
        Método para generar el gráfico de la matriz de confunción del modelo 
        Suport Vector Machine con Kernel lineal.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        # Se obtienen las predicciones.
        predictions = self.predict()
        
        # Se grafican los resultados.
        Graficos.plot_confusion_matrix(self.Y_test, predictions)
        
    # Método para generar el gráfico de importancias.
    def feature_importance(self, n_repeats=10, random_state=123, n_jobs=2):
        '''
        Método para generar el gráfico de importancia para las variables en el 
        modelo Suport Vector Machine con Kernel lineal.

        Parameters:
            None.
            
        Returns:
            n_repeats (int): Número de veces que se permuta una característica.
            random_state (int): Semiila para la reproducibilidad de los 
                                resultados.
            n_jobs (int): Número de trabajos en paralelo.
        '''
        # Se obtienen las importancias.
        result = permutation_importance(self.model, self.X, self.Y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        importances = result.importances_mean
        variables = self.X.columns
        
        # Se grafican los resultados.
        Graficos.feature_importance(importances, variables, 'Importancias caso Lineal')

# Se crea la clase para el modelo SVM con Kernel radial.
class RadialSVM(PreparacionDatos):
    # Se crea el constructor de la clase.
    def __init__(self, df, target_variable, param_grid=None, cv=3, scoring='accuracy', n_jobs=-1, verbose=0):
        '''
        Método contructor de la clase RadialSVM.

        Parameters:
            df (pandas.DataFrame): El dataframe o tabla con los datos a
                                   particionar.
            target_variable (string): EL nombre de la variable a predecir.
            param_grid (dict o list de dictionarios): Diccionario con los 
                                                      parametros.
            cv (int): Determina la estrategia de división de validación 
                      cruzada.
            scoring (string): Estrategia para evaluar el rendimiento del modelo 
                              de validación cruzada en el conjunto de pruebas. 
            n_jobs (int): Número de trabajos en paralelo.
            verbose (int): Controla la candidad de información desplegada.
            
        Returns:
            None.
        '''
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

    # Método para ajustar el modelo.
    def fit(self):
        '''
        Método para ajustar el modelo de Suport Vector Machine con Kernel 
        radial.

        Parameters:
            None.
            
        Returns:
            (pandas.DataFrame): El dataframe con los resultados del modelo.
        '''
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

    # Método para generar las prediciones del modelo.
    def predict(self):
        '''
        Método para generar la prediciones del modelo de Suport Vector Machine 
        con Kernel radial.

        Parameters:
            None.
            
        Returns:
            self.best_model.predict (ndarray): Arreglo con las predicicones.
        '''
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        return self.best_model.predict(self.X_test)

    # Método para obtener la precisión del modelo.
    def accuracy(self):
        '''
        Método para obtener la precisión del modelo de Suport Vector Machine 
        con Kernel radial.

        Parameters:
            None.
            
        Returns:
            accuracy (float): Nivel de precisión del modelo.
        '''
        # Se obtienen las predicciones.
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        predictions = self.predict()
        
        # Se obtiene la precisión.
        accuracy = accuracy_score(y_true=self.Y_test, y_pred=predictions, normalize=True)
        print(f"La precisión del test es: {100*accuracy}%")
        return accuracy
    
    # Método para generar el gráfico de la matriz de confunción.
    def plot_confusion_matrix(self):
        '''
        Método para generar el gráfico de la matriz de confunción del modelo 
        Suport Vector Machine con Kernel radial.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        # Se obtienen las predicciones.
        predictions = self.predict()
        
        # Se grafican los resultados.
        Graficos.plot_confusion_matrix(self.Y_test, predictions)

    # Método para generar el gráfico de importancias.
    def feature_importance(self, n_repeats=10, random_state=123, n_jobs=2):
        '''
        Método para generar el gráfico de importancia para las variables en el 
        modelo Suport Vector Machine con Kernel radial.

        Parameters:
            None.
            
        Returns:
            n_repeats (int): Número de veces que se permuta una característica.
            random_state (int): Semiila para la reproducibilidad de los 
                                resultados.
            n_jobs (int): Número de trabajos en paralelo.
        '''
        # Se obtienen las importancias.
        result = permutation_importance(self.best_model, self.X, self.Y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        importances = result.importances_mean
        variables = self.X.columns
        
        # Se grafican los resultados.
        Graficos.feature_importance(importances, variables, 'Importancias caso Radial')

# Se crea la clase para el modelo de Arboles de decisión.
class ArbolDeDecision(PreparacionDatos):
    # Se crea el constructor de la clase.
    def __init__(self, df, target_variable, param_grid=None, random_state=1000):
        '''
        Método contructor de la clase ArbolDeDecision.

        Parameters:
            df (pandas.DataFrame): El dataframe o tabla con los datos a
                                   particionar.
            target_variable (string): EL nombre de la variable a predecir.
            param_grid (dict o list de dictionarios): Diccionario con los 
                                                      parametros.
            random_state (int): Semiila para la reproducibilidad de los 
                                resultados.
            
        Returns:
            None.
        '''
        PreparacionDatos.__init__(self, df, target_variable)
        self.extraer_variables()
        self.X = (self.X - np.min(self.X)) / (np.max(self.X - np.min(self.X)))
        self.dividir_datos()
        self.dt = DecisionTreeClassifier()

    # Método para ajustar el modelo.
    def fit(self):
        
        '''
        Método que entrena el modelo de árbol de decisión y calcula la 
        exactitud y la importancia de las características, las guarda en los 
        args inicializados en el init.
        
        Parameters
        ----------
        
        Returns
        -------
        
        '''
        
        self.dt.fit(self.X_train, self.Y_train)
    
    # Método para generar las prediciones del modelo.
    def predict(self):
        '''
        Método para generar la prediciones del modelo de Arboles de decision

        Parameters:
            None.
            
        Returns:
            self.model.predict (ndarray): Arreglo con las predicicones.
        '''
        return self.dt.predict(self.X_test)
    
    # Método para sacar la precisión del modelo. 
    def accuracy(self):
        '''
        Método para obtener la precisión del modelo de arboles de decision

        Parameters:
            None.
            
        Returns:
            accuracy (float): Nivel de precisión del modelo.
        '''
        # Se obtiene el nivel de precisión del modelo.
        accuracy = self.dt.score(self.X_test, self.Y_test)
        
        
        print(f"La precisión del test es: {100*accuracy}%")
        return accuracy
    
    # Método para generar el gráfico de la matriz de confunción.
    def plot_confusion_matrix(self):
        '''
        Método para generar el gráfico de la matriz de confunción del modelo 
        Arboles de decisión

        Parameters:
            None.
            
        Returns:
            None.
        '''
        predictions = self.predict()
        Graficos.plot_confusion_matrix(self.Y_test, predictions)

    # Método para generar el gráfico de importancias.
    def feature_importance(self, n_repeats=10, random_state=123, n_jobs=2):
        result = permutation_importance(self.dt, self.X_test, self.Y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        importances = result.importances_mean
        variables = self.X.columns

        Graficos.feature_importance(importances, variables, 'Importancias caso Arboles de decisión')
    
    def plot_tree(self):
        plt.figure(figsize=(400, 200))
        plot_tree(self.dt, feature_names=self.df.columns[:-1], class_names=['Non-Diabetic', 'Diabetic'], filled=True)
        plt.show()
        
    
# Se crea la clase para el modelo de bosques aleatorios.
class RandomForestModel(PreparacionDatos):
    # Se crea el constructor de la clase.
    def __init__(self, df, target_variable, param_grid=None, random_state=1000):
        '''
        Método contructor de la clase RandomForestModel.

        Parameters:
            df (pandas.DataFrame): El dataframe o tabla con los datos a
                                   particionar.
            target_variable (string): EL nombre de la variable a predecir.
            param_grid (dict o list de dictionarios): Diccionario con los 
                                                      parametros.
            random_state (int): Semiila para la reproducibilidad de los 
                                resultados.
            
        Returns:
            None.
        '''
        PreparacionDatos.__init__(self, df, target_variable)
        if param_grid is None:
            param_grid = {
                'n_estimators': list(range(20,30)),
                'min_samples_leaf': list(range(1,15))
            }
        self.param_grid = param_grid
        self.random_state = random_state
        self.best_model = None
        self.extraer_variables()
        self.dividir_datos()

    # Método para ajustar el modelo.
    def fit(self):
        '''
        Método para ajustar el modelo Random Forest.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        # Se incializan las variables que contendrán los mejores resultados.
        best_score = 0
        best_params = {}
        for n in self.param_grid['n_estimators']:
            for m in self.param_grid['min_samples_leaf']:
                model = RandomForestClassifier(n_estimators=n, min_samples_leaf=m, random_state=self.random_state)
                model.fit(self.X_train, self.Y_train)
                score = model.score(self.X_test, self.Y_test)
                if score > best_score:
                    best_score = score
                    best_params = {'n_estimators': n, 'min_samples_leaf': m}
        
        # Se imprimen los mejores resultados.
        print("Mejor precisión encontrada:", best_score)
        print("Mejores hiperparámetros encontrados:", best_params)
        
        # Se guardan estos resultados.
        self.best_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], 
                                                 min_samples_leaf=best_params['min_samples_leaf'], 
                                                 random_state=self.random_state)
        self.best_model.fit(self.X_train, self.Y_train)

    # Método para generar las predicciones.
    def predict(self):
        '''
        Método para generar la prediciones del modelo Random Forest.

        Parameters:
            None.
            
        Returns:
            self.best_model.predict (ndarray): Arreglo con las predicicones.
        '''
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        return self.best_model.predict(self.X_test)
    
    # Método para obtener la precisión del modelo.
    def accuracy(self):
        '''
        Método para obtener la precisión del modelo Random Forest.

        Parameters:
            None.
            
        Returns:
            accuracy (float): Nivel de precisión del modelo.
        '''
        # Se obtienen las predicciones.
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama al método fit primero.")
        predictions = self.predict()
        
        # Se obtiene la precisión.
        accuracy = accuracy_score(y_true=self.Y_test, y_pred=predictions, normalize=True)
        print(f"La precisión del test es: {100*accuracy}%")
        return accuracy
    
    # Método para generar el gráfico de la matriz de confunción.
    def plot_confusion_matrix(self):
        '''
        Método para generar el gráfico de la matriz de confunción del modelo 
        Random Forest.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        # Se obtienen las predicciones.
        predictions = self.predict()
        
        # Se grafican los resultados.
        Graficos.plot_confusion_matrix(self.Y_test, predictions)

    # Método para generar el gráfico de importancias.
    def feature_importance(self, n_repeats=10, random_state=123, n_jobs=2):
        '''
        Método para generar el gráfico de importancia para las variables en el 
        modelo Random Forest.

        Parameters:
            None.
            
        Returns:
            n_repeats (int): Número de veces que se permuta una característica.
            random_state (int): Semiila para la reproducibilidad de los 
                                resultados.
            n_jobs (int): Número de trabajos en paralelo.
        '''
        # Se obtienen las importancias.
        result = permutation_importance(self.best_model, self.X, self.Y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        importances = result.importances_mean
        variables = self.X.columns
        
        # Se grafican los resultados.
        Graficos.feature_importance(importances, variables, 'Importancias caso Random Forest')
