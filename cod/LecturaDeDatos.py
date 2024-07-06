import pandas as pd

class LecturaDeDatos:
    def __init__(self, url):
        self.url = url
        self.df = None

    def descargar_datos(self):
        self.df = pd.read_csv(self.url)
        print("Datos descargados exitosamente.")
    
    def describir_datos(self):
        if self.df is not None:
            descripcion = self.df.describe().round()
            print("Descripción de los datos:")
            print(descripcion)
            return descripcion
        else:
            print("Primero debe descargar los datos.")
    
    def informacion_datos(self):
        if self.df is not None:
            print("Información de los datos:")
            self.df.info()
        else:
            print("Primero debe descargar los datos.")
