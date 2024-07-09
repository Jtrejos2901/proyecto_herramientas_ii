import pandas as pd

class LecturaDeDatos:
    def __init__(self, url):
        '''
        Constructor de la clase LecturaDeDatos.

        Parameters
        ----------
        url : str
            La URL desde donde se cargarán los datos.
        '''
        self.url = url
        self.df = None
    
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
    def url(self):
        '''
        Obtiene la url actual.

        Returns
        -------
        str:
             url.
        '''
        return self.url

    @df.setter
    def url(self, nueva_url):
        '''Establece una nueva url.

        Parameters
        ----------
        nueva_url : str
            La nueva url.

        Returns
        -------
        None
        '''
        self.df = nueva_url
    def __str__(self):
        '''Devuelve una representación en cadena del DataFrame.

        Returns
        -------
        str
            Cadena que representa el DataFrame y url.
        '''
        return f'El dataframe dado es: {self.df} \n y la url es {self.url}'
    

    def descargar_datos(self):
        '''
        Descarga los datos desde la URL proporcionada y los carga en un DataFrame.

        Returns
        -------
        None
        '''
        self.df = pd.read_csv(self.url)
        print("Datos descargados exitosamente.")
    
    def describir_datos(self):
        '''
        Genera y muestra una descripción estadística del DataFrame si los datos están disponibles.

        Returns
        -------
        pandas.DataFrame or None
            La descripción estadística de los datos si están disponibles, None si no se han descargado los datos.
        '''
        if self.df is not None:
            descripcion = self.df.describe()
            print("Descripción de los datos:")
            return descripcion
        else:
            print("Primero debe descargar los datos.")
    
    def informacion_datos(self):
        '''
        Muestra información detallada del DataFrame si los datos están disponibles.

        Returns
        -------
        None
        '''
        if self.df is not None:
            print("Información de los datos:")
            self.df.info()
        else:
            print("Primero debe descargar los datos.")

