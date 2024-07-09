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
    
    @property
    def df(self):
        '''Obtiene el DataFrame actual.

        Returns
        -------
        pandas.DataFrame
            El DataFrame actual.
        '''
        return self.df

    @df.setter
    def df(self, nuevo_df):
        '''Establece un nuevo DataFrame.

        Parameters
        ----------
        nuevo_df : pandas.DataFrame
            El nuevo DataFrame a establecer.

        Returns
        -------
        None
        '''
        self.df = nuevo_df

    @property
    def target_variable(self):
        '''Obtiene la variable objetivo actual.

        Returns
        -------
        str
            La variable objetivo actual.
        '''
        return self.target_variable

    @target_variable.setter
    def target_variable(self, nueva_target_variable):
        '''Establece una nueva variable objetivo.

        Parameters
        ----------
        nueva_target_variable : str
            La nueva variable objetivo a establecer.

        Returns
        -------
        None
        '''
        self.target_variable = nueva_target_variable

    @property
    def X(self):
        '''Obtiene los datos de entrada X.

        Returns
        -------
        pandas.DataFrame
            Los datos de entrada X.
        '''
        return self.X

    @X.setter
    def X(self, nuevo_X):
        '''Establece nuevos datos de entrada X.

        Parameters
        ----------
        nuevo_X : pandas.DataFrame
            Los nuevos datos de entrada X a establecer.

        Returns
        -------
        None
        '''
        self.X = nuevo_X

    @property
    def Y(self):
        '''Obtiene los datos de salida Y.

        Returns
        -------
        pandas.DataFrame
            Los datos de salida Y.
        '''
        return self.Y

    @Y.setter
    def Y(self, nuevo_Y):
        '''Establece nuevos datos de salida Y.

        Parameters
        ----------
        nuevo_Y : pandas.DataFrame
            Los nuevos datos de salida Y a establecer.

        Returns
        -------
        None
        '''
        self.Y = nuevo_Y

    @property
    def X_train(self):
        '''Obtiene los datos de entrenamiento X.

        Returns
        -------
        pandas.DataFrame
            Los datos de entrenamiento X.
        '''
        return self.X_train

    @X_train.setter
    def X_train(self, nuevo_X_train):
        '''Establece nuevos datos de entrenamiento X.

        Parameters
        ----------
        nuevo_X_train : pandas.DataFrame
            Los nuevos datos de entrenamiento X a establecer.

        Returns
        -------
        None
        '''
        self.X_train = nuevo_X_train

    @property
    def X_test(self):
        '''Obtiene los datos de prueba X.

        Returns
        -------
        pandas.DataFrame
            Los datos de prueba X.
        '''
        return self.X_test

    @X_test.setter
    def X_test(self, nuevo_X_test):
        '''Establece nuevos datos de prueba X.

        Parameters
        ----------
        nuevo_X_test : pandas.DataFrame
            Los nuevos datos de prueba X a establecer.

        Returns
        -------
        None
        '''
        self.X_test = nuevo_X_test

    @property
    def Y_train(self):
        '''Obtiene los datos de entrenamiento Y.

        Returns
        -------
        pandas.DataFrame
            Los datos de entrenamiento Y.
        '''
        return self.Y_train

    @Y_train.setter
    def Y_train(self, nuevo_Y_train):
        '''Establece nuevos datos de entrenamiento Y.

        Parameters
        ----------
        nuevo_Y_train : pandas.DataFrame
            Los nuevos datos de entrenamiento Y a establecer.

        Returns
        -------
        None
        '''
        self.Y_train = nuevo_Y_train

    @property
    def Y_test(self):
        '''Obtiene los datos de prueba Y.

        Returns
        -------
        pandas.DataFrame
            Los datos de prueba Y.
        '''
        return self.Y_test

    @Y_test.setter
    def Y_test(self, nuevo_Y_test):
        '''Establece nuevos datos de prueba Y.

        Parameters
        ----------
        nuevo_Y_test : pandas.DataFrame
            Los nuevos datos de prueba Y a establecer.

        Returns
        -------
        None
        '''
        self.Y_test = nuevo_Y_test

    def __str__(self):
        '''Devuelve una representación en cadena del estado del objeto.

        Returns
        -------
        str
            Cadena que representa el estado del objeto.
        '''
        return (f'DataFrame: {self.df}\n'
                f'Variable objetivo: {self.target_variable}\n'
                f'X: {self.X}\n'
                f'Y: {self.Y}\n'
                f'X_train: {self.X_train}\n'
                f'X_test: {self.X_test}\n'
                f'Y_train: {self.Y_train}\n'
                f'Y_test: {self.Y_test}')

    # Metodo para la extracción de variables.
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
    
    @property
    def model(self):
        '''Obtiene el modelo SVM lineal.

        Returns
        -------
        sklearn.svm.SVC
            El modelo SVM lineal.
        '''
        return self.model

    @model.setter
    def model(self, nuevo_model):
        '''Establece un nuevo modelo SVM lineal.

        Parameters
        ----------
        nuevo_model : sklearn.svm.SVC
            El nuevo modelo SVM lineal a establecer.

        Returns
        -------
        None
        '''
        self.model = nuevo_model

    def __str__(self):
        '''Devuelve una representación en cadena del estado del objeto LinearSVM.

        Returns
        -------
        str
            Cadena que representa el estado del objeto LinearSVM.
        '''
        return (f'DataFrame: {self.df}\n'
                f'Variable objetivo: {self.target_variable}\n'
                f'X: {self.X}\n'
                f'Y: {self.Y}\n'
                f'X_train: {self.X_train}\n'
                f'X_test: {self.X_test}\n'
                f'Y_train: {self.Y_train}\n'
                f'Y_test: {self.Y_test}\n'
                f'Modelo SVM: {self.model}')
    
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
    def __init__(self, df, target_variable, param_grid=None, cv=100, scoring='accuracy', n_jobs=-1, verbose=0):
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

    @property
    def grid_search(self):
        '''Obtiene el objeto GridSearchCV.

        Returns
        -------
        sklearn.model_selection.GridSearchCV
            El objeto GridSearchCV.
        '''
        return self.grid_search

    @grid_search.setter
    def grid_search(self, nuevo_grid_search):
        '''Establece un nuevo objeto GridSearchCV.

        Parameters
        ----------
        nuevo_grid_search : sklearn.model_selection.GridSearchCV
            El nuevo objeto GridSearchCV a establecer.

        Returns
        -------
        None
        '''
        self.grid_search = nuevo_grid_search

    @property
    def best_model(self):
        '''Obtiene el mejor modelo encontrado por GridSearchCV.

        Returns
        -------
        sklearn.svm.SVC
            El mejor modelo SVM encontrado por GridSearchCV.
        '''
        return self.best_model

    @best_model.setter
    def best_model(self, nuevo_best_model):
        '''Establece el mejor modelo encontrado por GridSearchCV.

        Parameters
        ----------
        nuevo_best_model : sklearn.svm.SVC
            El nuevo mejor modelo SVM a establecer.

        Returns
        -------
        None
        '''
        self.best_model = nuevo_best_model

    def __str__(self):
        '''Devuelve una representación en cadena del estado del objeto RadialSVM.

        Returns
        -------
        str
            Cadena que representa el estado del objeto RadialSVM.
        '''
        return (f'DataFrame: {self.df}\n'
                f'Variable objetivo: {self.target_variable}\n'
                f'X: {self.X}\n'
                f'Y: {self.Y}\n'
                f'X_train: {self.X_train}\n'
                f'X_test: {self.X_test}\n'
                f'Y_train: {self.Y_train}\n'
                f'Y_test: {self.Y_test}\n'
                f'GridSearchCV: {self.grid_search}\n'
                f'Mejor Modelo: {self.best_model}')
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
    
    def __init__(self, df, target_variable):
        """
        Inicializa la clase DecisionTrees.

        Parámetros:
        -----------
        df : DataFrame
            El DataFrame que contiene los datos.
        target_variable : str
            El nombre de la variable objetivo en el DataFrame.
        """
        self.df = df
        self.target_variable = target_variable
        self.x_data = df.drop([target_variable], axis=1)
        self.y = df[target_variable].values
        self.x = (self.x_data - np.min(self.x_data)) / (np.max(self.x_data) - np.min(self.x_data))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=73)
        self.dt = DecisionTreeClassifier(random_state=73)
        self.feature_importance_df = None
        self.accuracy_score = None

    @property
    def df(self):
        '''Obtiene el DataFrame actual.

        Returns
        -------
        pandas.DataFrame
            El DataFrame actual.
        '''
        return self.df

    @df.setter
    def df(self, nuevo_df):
        '''Establece un nuevo DataFrame.

        Parameters
        ----------
        nuevo_df : pandas.DataFrame
            El nuevo DataFrame a establecer.

        Returns
        -------
        None
        '''
        self.df = nuevo_df

    @property
    def target_variable(self):
        '''Obtiene la variable objetivo actual.

        Returns
        -------
        str
            La variable objetivo actual.
        '''
        return self.target_variable

    @target_variable.setter
    def target_variable(self, nueva_target_variable):
        '''Establece una nueva variable objetivo.

        Parameters
        ----------
        nueva_target_variable : str
            La nueva variable objetivo a establecer.

        Returns
        -------
        None
        '''
        self.target_variable = nueva_target_variable

    @property
    def x_data(self):
        '''Obtiene los datos de características (X) excluyendo la variable objetivo.

        Returns
        -------
        pandas.DataFrame
            Los datos de características (X).
        '''
        return self.x_data

    @x_data.setter
    def x_data(self, nuevo_x_data):
        '''Establece nuevos datos de características (X) excluyendo la variable objetivo.

        Parameters
        ----------
        nuevo_x_data : pandas.DataFrame
            Los nuevos datos de características (X) a establecer.

        Returns
        -------
        None
        '''
        self.x_data = nuevo_x_data

    @property
    def y(self):
        '''Obtiene los datos de la variable objetivo (Y).

        Returns
        -------
        numpy.ndarray
            Los datos de la variable objetivo (Y).
        '''
        return self.y

    @y.setter
    def y(self, nuevo_y):
        '''Establece nuevos datos de la variable objetivo (Y).

        Parameters
        ----------
        nuevo_y : numpy.ndarray
            Los nuevos datos de la variable objetivo (Y) a establecer.

        Returns
        -------
        None
        '''
        self.y = nuevo_y

    @property
    def x(self):
        '''Obtiene los datos de características (X) normalizados.

        Returns
        -------
        pandas.DataFrame
            Los datos de características (X) normalizados.
        '''
        return self.x

    @x.setter
    def x(self, nuevo_x):
        '''Establece nuevos datos de características (X) normalizados.

        Parameters
        ----------
        nuevo_x : pandas.DataFrame
            Los nuevos datos de características (X) normalizados a establecer.

        Returns
        -------
        None
        '''
        self.x = nuevo_x

    @property
    def dt(self):
        '''Obtiene el clasificador de árbol de decisión.

        Returns
        -------
        sklearn.tree.DecisionTreeClassifier
            El clasificador de árbol de decisión.
        '''
        return self.dt

    @dt.setter
    def dt(self, nuevo_dt):
        '''Establece un nuevo clasificador de árbol de decisión.

        Parameters
        ----------
        nuevo_dt : sklearn.tree.DecisionTreeClassifier
            El nuevo clasificador de árbol de decisión a establecer.

        Returns
        -------
        None
        '''
        self.dt = nuevo_dt

    @property
    def feature_importance_df(self):
        '''Obtiene el DataFrame de la importancia de las características.

        Returns
        -------
        pandas.DataFrame
            El DataFrame de la importancia de las características.
        '''
        return self.feature_importance_df

    @feature_importance_df.setter
    def feature_importance_df(self, nuevo_feature_importance_df):
        '''Establece un nuevo DataFrame de la importancia de las características.

        Parameters
        ----------
        nuevo_feature_importance_df : pandas.DataFrame
            El nuevo DataFrame de la importancia de las características a establecer.

        Returns
        -------
        None
        '''
        self.feature_importance_df = nuevo_feature_importance_df

    @property
    def accuracy_score(self):
        '''Obtiene la puntuación de precisión del modelo.

        Returns
        -------
        float
            La puntuación de precisión del modelo.
        '''
        return self.accuracy_score

    @accuracy_score.setter
    def accuracy_score(self, nuevo_accuracy_score):
        '''Establece una nueva puntuación de precisión del modelo.

        Parameters
        ----------
        nuevo_accuracy_score : float
            La nueva puntuación de precisión del modelo a establecer.

        Returns
        -------
        None
        '''
        self.accuracy_score = nuevo_accuracy_score

    def __str__(self):
        '''Devuelve una representación en cadena del estado del objeto ArbolDeDecision.

        Returns
        -------
        str
            Cadena que representa el estado del objeto ArbolDeDecision.
        '''
        return (f'DataFrame: {self.df}\n'
                f'Variable objetivo: {self.target_variable}\n'
                f'X_data: {self.x_data}\n'
                f'Y: {self.y}\n'
                f'X: {self.x}\n'
                f'X_train: {self.x_train}\n'
                f'X_test: {self.x_test}\n'
                f'Y_train: {self.y_train}\n'
                f'Y_test: {self.y_test}\n'
                f'Clasificador de Árbol de Decisión: {self.dt}\n'
                f'Importancia de Características: {self.feature_importance_df}\n'
                f'Puntuación de Precisión: {self.accuracy_score}')
    def fit(self):
        """
        Entrena el modelo de árbol de decisión utilizando GridSearchCV 
        para encontrar los mejores hiperparámetros.
        
        Parameters:
            None.
            
        Returns:
            None.
        
        """
        param_grid = {
            'max_depth': [3, 5, 7, 8, 10, 13, 15],
            'min_samples_split': [2, 5, 7, 10, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'splitter': ['best']
        }
        grid_search = GridSearchCV(self.dt, param_grid, cv=100, scoring='accuracy')
        grid_search.fit(self.x_train, self.y_train)
        self.dt = grid_search.best_estimator_
        self.accuracy_score = self.dt.score(self.x_test, self.y_test)

        feature_importances = self.dt.feature_importances_
        self.feature_importance_df = pd.DataFrame({
            'Feature': self.x_data.columns,
            'Importance': feature_importances
        })

    def accuracy(self):
        """
        Devuelve la precisión del modelo.

        Retorno:
        --------
        float
            La precisión del modelo.
        """
        return self.accuracy_score

    def plot_confusion_matrix(self):
        """
        Genera un gráfico de la matriz de confusión.
        
        Parameters:
            None.
            
        Returns:
            None.
        """
        y_pred = self.dt.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 6))
        plot_confusion_matrix(cm, figsize=(10, 6), cmap=plt.cm.Blues, show_absolute=True, show_normed=True)
        plt.title('Matriz de Confusión')
        plt.show()

    def feature_importance(self):
        """
        Muestra y grafica las importancias de las características.
        
        Parameters:
            None.
            
        Returns:
            None.
        """
        print("Importancias de las características:")
        for feature, importance in zip(self.feature_importance_df['Feature'], self.feature_importance_df['Importance']):
            print(f"{feature}: {importance:.3f}")
        
        plt.figure(figsize=(10, 6))
        plt.barh(self.feature_importance_df['Feature'], self.feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.title('Importancia de las Características')
        plt.tight_layout()
        plt.show()

    def plot_tree(self):
        """
        Genera una visualización del árbol de decisión.
        Parameters:
            None.
            
        Returns:
            None.
        """
        plt.figure(figsize=(160, 80))
        plot_tree(self.dt, feature_names=self.x_data.columns, class_names=['Non-Diabetic', 'Diabetic'], filled=True)
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

    @property
    def param_grid(self):
        '''Obtiene el diccionario de parámetros para GridSearchCV.

        Returns
        -------
        dict
            El diccionario de parámetros.
        '''
        return self.param_grid

    @param_grid.setter
    def param_grid(self, nuevo_param_grid):
        '''Establece un nuevo diccionario de parámetros para GridSearchCV.

        Parameters
        ----------
        nuevo_param_grid : dict
            El nuevo diccionario de parámetros a establecer.

        Returns
        -------
        None
        '''
        self.param_grid = nuevo_param_grid

    @property
    def random_state(self):
        '''Obtiene el estado aleatorio utilizado para la reproducibilidad.

        Returns
        -------
        int
            El estado aleatorio.
        '''
        return self.random_state

    @random_state.setter
    def random_state(self, nuevo_random_state):
        '''Establece un nuevo estado aleatorio para la reproducibilidad.

        Parameters
        ----------
        nuevo_random_state : int
            El nuevo estado aleatorio a establecer.

        Returns
        -------
        None
        '''
        self.random_state = nuevo_random_state

    @property
    def best_model(self):
        '''Obtiene el mejor modelo encontrado por GridSearchCV.

        Returns
        -------
        sklearn.ensemble.RandomForestClassifier
            El mejor modelo RandomForest encontrado por GridSearchCV.
        '''
        return self.best_model

    @best_model.setter
    def best_model(self, nuevo_best_model):
        '''Establece el mejor modelo encontrado por GridSearchCV.

        Parameters
        ----------
        nuevo_best_model : sklearn.ensemble.RandomForestClassifier
            El nuevo mejor modelo RandomForest a establecer.

        Returns
        -------
        None
        '''
        self.best_model = nuevo_best_model

    def __str__(self):
        '''Devuelve una representación en cadena del estado del objeto RandomForestModel.

        Returns
        -------
        str
            Cadena que representa el estado del objeto RandomForestModel.
        '''
        return (f'DataFrame: {self.df}\n'
                f'Variable objetivo: {self.target_variable}\n'
                f'X: {self.X}\n'
                f'Y: {self.Y}\n'
                f'X_train: {self.X_train}\n'
                f'X_test: {self.X_test}\n'
                f'Y_train: {self.Y_train}\n'
                f'Y_test: {self.Y_test}\n'
                f'Parámetros de GridSearchCV: {self.param_grid}\n'
                f'Estado Aleatorio: {self.random_state}\n'
                f'Mejor Modelo: {self.best_model}')
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


    
        