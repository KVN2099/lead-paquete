from IPython.display import display
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prince import PCA as PCA_Prince
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Import KMedoids conditionally to handle environments without scikit-learn-extra
try:
    from sklearn_extra.cluster import KMedoids
    SKLEARN_EXTRA_AVAILABLE = True
except ImportError:
    SKLEARN_EXTRA_AVAILABLE = False
    
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from math import ceil, pi, floor
from seaborn import color_palette
from scipy.cluster.hierarchy import dendrogram, ward, average, single, complete, fcluster, linkage
from sklearn.manifold import TSNE
# Import UMAP conditionally
try:
    import umap.umap_ as um
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import statistics 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class AnalisisDatosExploratorio:
    def __init__(self, file_path, delimiter=';', decimal='.', tipo_modelo='no_supervisado'):
        self.file_path = file_path
        self.delimiter = delimiter
        self.decimal = decimal
        self.tipo_modelo = tipo_modelo
        self.data = None
        self.df_normalizada = None
        self.__cargar_datos()

    def __cargar_datos(self):
        """
        Carga los datos desde un archivo CSV, procesándolos de acuerdo al tipo de modelo.
        """
        try:
            # Cargar el archivo CSV con tipos de datos inferidos
            self.data = pd.read_csv(self.file_path, delimiter=self.delimiter, decimal=self.decimal)
            
            # Eliminar filas con valores nulos
            self.data = self.data.dropna()
            
            # Convertir columnas categóricas a numéricas si es necesario
            if self.tipo_modelo == 'no_supervisado':
                # Identificar columnas categóricas
                categorical_columns = self.data.select_dtypes(include=['object']).columns
                
                # Convertir cada columna categórica a numérica usando one-hot encoding
                for col in categorical_columns:
                    # Crear variables dummy y agregarlas al DataFrame
                    dummies = pd.get_dummies(self.data[col], prefix=col)
                    self.data = pd.concat([self.data, dummies], axis=1)
                    # Eliminar la columna original
                    self.data = self.data.drop(col, axis=1)
                
                print("\nInformación del dataset después del preprocesamiento:")
                print(f"Dimensiones: {self.data.shape}")
                print("\nTipos de datos de las columnas:")
                print(self.data.dtypes)
            
        except Exception as e:
            print(f"Error al cargar los datos: {str(e)}")
            self.data = None

    def resumen_general(self):
        """
        Imprime información general y estadísticas descriptivas del dataset.
        """
        print("Información del dataset:")
        print(self.data.info())

        print("\nEstadísticas descriptivas:")
        print(self.data.describe())

    def analisisNumerico(self):
        """
        Filtra y devuelve solo las columnas numéricas del dataset.
        """
        return self.data.select_dtypes(include=["number"])

    def normalizar_datos(self):
        """
        Normaliza los datos usando Z-score y maneja variables categóricas.
        """
        if self.data is None or self.data.empty:
            print("Error: No hay datos para normalizar.")
            return None
            
        # Separar variables numéricas y categóricas
        numericas = self.data.select_dtypes(include=['int64', 'float64'])
        categoricas = self.data.select_dtypes(exclude=['int64', 'float64'])
        
        # Verificar si hay columnas numéricas
        if numericas.empty:
            print("Advertencia: No se encontraron columnas numéricas para normalizar.")
            self.df_normalizada = pd.get_dummies(categoricas) if not categoricas.empty else None
            return self.df_normalizada
            
        # Imprimir información sobre las columnas
        print("\nColumnas numéricas encontradas:")
        print(numericas.columns.tolist())
        print("\nColumnas categóricas encontradas:")
        print(categoricas.columns.tolist())
        
        # Normalizar variables numéricas
        scaler = StandardScaler()
        datos_normalizados = scaler.fit_transform(numericas)
        df_normalizado = pd.DataFrame(datos_normalizados, columns=numericas.columns, index=numericas.index)
        
        # Manejar variables categóricas
        if not categoricas.empty:
            try:
                df_dummies = pd.get_dummies(categoricas)
                print(f"\nVariables dummy creadas: {df_dummies.shape[1]}")
                # Concatenar variables numéricas normalizadas y categóricas codificadas
                self.df_normalizada = pd.concat([df_normalizado, df_dummies], axis=1)
            except Exception as e:
                print(f"Error al procesar variables categóricas: {str(e)}")
                self.df_normalizada = df_normalizado
        else:
            self.df_normalizada = df_normalizado
        
        print(f"\nDimensiones del DataFrame normalizado: {self.df_normalizada.shape}")
        return self.df_normalizada

    def ResVarPred(self, VarPred):
        """
        Calcula el valor máximo, mínimo y los cuartiles de una variable objetivo.
        """
        Cuartil = statistics.quantiles(VarPred)
        val_max = np.max(VarPred)
        val_min = np.min(VarPred)
        return {
            "Máximo": val_max,
            "Cuartiles": Cuartil,
            "Mínimo": val_min
        }

    def graficar_relaciones(self):
        """
        Graficar relaciones entre variables numéricas.
        """
        # Seleccionar solo columnas numéricas
        numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        print("Columnas numéricas disponibles:", numeric_data.columns.tolist())
        print("Número de columnas numéricas:", len(numeric_data.columns))
        
        # Crear el mapa de calor de correlaciones
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Mapa de Correlación')
        plt.show()

        # Crear subplots solo si las columnas necesarias están presentes
        if all(col in numeric_data.columns for col in ['precio', 'tamaño_motor', 'kilometraje', 'año']):
            f, axes = plt.subplots(2, 2, figsize=(12, 8))
            sns.regplot(x='precio', y='tamaño_motor', data=numeric_data, ax=axes[0, 0], scatter_kws={'alpha': 0.6})
            axes[0, 0].set_xlabel('Precio', fontsize=14)
            axes[0, 0].set_ylabel('Tamaño del motor', fontsize=14)

            sns.regplot(x='precio', y='kilometraje', data=numeric_data, ax=axes[0, 1], scatter_kws={'alpha': 0.6})
            axes[0, 1].set_xlabel('Precio', fontsize=14)
            axes[0, 1].set_ylabel('Kilometraje', fontsize=14)

            sns.regplot(x='precio', y='año', data=numeric_data, ax=axes[1, 0], scatter_kws={'alpha': 0.6})
            axes[1, 0].set_xlabel('Precio', fontsize=14)
            axes[1, 0].set_ylabel('Año', fontsize=14)

            axes[1, 1].axis('off')
            plt.show()
        else:
            print("Algunas columnas numéricas requeridas no están disponibles en el dataset.")

    def analisis(self):
        """
        Realiza un análisis completo de las variables del DataFrame.
        """
        print("Dimensiones:", self.data.shape)
        print(self.data.head())
        print(self.data.describe())
        print("Resumen de datos sin valores nulos:\n", self.data.dropna().describe())
        print("Media:", self.data.mean(numeric_only=True))
        print("Mediana:", self.data.median(numeric_only=True))
        print("Desviación estándar:", self.data.std(numeric_only=True, ddof=0))
        print("Valor máximo:", self.data.max(numeric_only=True))
        print("Valor mínimo:", self.data.min(numeric_only=True))
        print("Cuantiles:\n", self.data.quantile(np.array([0, .33, .50, .75, 1]), numeric_only=True))
        
        self.__graficosBoxplot()
        self.__funcionDensidad()
        self.__histograma()
        self.__graficoDeCorrelacion()

    def __graficosBoxplot(self):
        """
        Grafica un boxplot de las variables numéricas.
        """
        self.data.boxplot(figsize=(15, 8), grid=False, vert=False)
        plt.title("Boxplot de las Variables Numéricas")
        plt.show()

    def __funcionDensidad(self):
        """
        Grafica la función de densidad de las variables numéricas.
        """
        self.data.plot(kind='density', figsize=(12, 8), linewidth=2)
        plt.title("Función de Densidad de las Variables Numéricas")
        plt.show()

    def __histograma(self):
        """
        Grafica el histograma de las variables numéricas.
        """
        self.data.hist(figsize=(10, 6), bins=20, edgecolor="black", grid=False)
        plt.suptitle("Histograma de Variables Numéricas")
        plt.show()

    def __graficoDeCorrelacion(self):
        """
        Graficar el mapa de correlación entre las variables numéricas.
        """
        corr = self.data.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", square=True, vmin=-1, vmax=1, fmt=".2f")
        plt.title("Mapa de Calor de Correlación")
        plt.show()

    def __str__(self):
        return f'AnalisisDatosExploratorio con DataFrame de dimensiones: {self.data.shape}'
    

class NoSupervisados(AnalisisDatosExploratorio):
    def __init__(self, path=None, num=None, df=None):
        if df is not None:
            self.df_normalizada = df  # Usar el nombre correcto del atributo
        else:
            super().__init__(path, num)  # Llama al constructor de la clase base
    
    def verificar_datos(self):
        """
        Verifica que los datos estén preparados para el análisis.
        """
        if not hasattr(self, 'df_normalizada') or self.df_normalizada is None or self.df_normalizada.empty:
            print("Error: No hay datos normalizados disponibles.")
            return False
        return True

    def HAC(self, num_clusters=3, metodo='ward', criterio='maxclust'):
        """
        Realiza un Análisis Jerárquico Aglomerativo (HAC) sobre los datos normalizados.
        """
        if not self.verificar_datos():
            return None
            
        try:
            # Calcular la matriz de enlaces
            if metodo == 'ward':
                Z = ward(self.df_normalizada)
            elif metodo == 'average':
                Z = average(self.df_normalizada)
            elif metodo == 'complete':
                Z = complete(self.df_normalizada)
            elif metodo == 'single':
                Z = single(self.df_normalizada)
            else:
                print(f"Método {metodo} no reconocido. Usando 'ward' por defecto.")
                Z = ward(self.df_normalizada)
            
            # Obtener clusters
            clusters = fcluster(Z, num_clusters, criterion=criterio)
            
            # Reducción de dimensionalidad para visualización
            pca = PCA(n_components=2)
            coords = pca.fit_transform(self.df_normalizada)
            
            # Visualizar clusters en el espacio PCA
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                                c=clusters, 
                                cmap='viridis',
                                alpha=0.6)
            
            plt.title(f'Clusters usando HAC ({metodo.upper()})', pad=20, fontsize=14)
            plt.xlabel('Primera Componente Principal', fontsize=12)
            plt.ylabel('Segunda Componente Principal', fontsize=12)
            plt.colorbar(scatter, label='Cluster')
            
            # Añadir grid y ajustar layout
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return clusters
            
        except Exception as e:
            print(f"Error durante el clustering jerárquico: {str(e)}")
            return None

    def ACP(self, n_componentes=5):
        """
        Realiza el Análisis de Componentes Principales (ACP) utilizando scikit-learn.
        """
        if not self.verificar_datos():
            return None
            
        try:
            # Realizar el ACP
            self.__modelo_acp = PCA(n_components=n_componentes)
            componentes = self.__modelo_acp.fit_transform(self.df_normalizada)
            
            # Obtener coordenadas de los individuos
            self.coordenadas_ind = pd.DataFrame(
                componentes,
                index=self.df_normalizada.index,
                columns=[f'PC{i+1}' for i in range(n_componentes)]
            )
            
            # Calcular correlaciones de las variables con los componentes
            # Multiplicar los loadings por la raíz cuadrada de los valores propios
            loadings = self.__modelo_acp.components_.T
            sqrt_eigenvalues = np.sqrt(self.__modelo_acp.explained_variance_)
            correlaciones = loadings * sqrt_eigenvalues
            
            self.correlacion_var = pd.DataFrame(
                correlaciones,
                columns=[f'PC{i+1}' for i in range(n_componentes)],
                index=self.df_normalizada.columns
            )
            
            # Obtener varianza explicada
            self.var_explicada = self.__modelo_acp.explained_variance_ratio_
            
            # Mostrar resultados
            print(f"\nVarianza explicada por componente:")
            for i, var in enumerate(self.var_explicada):
                print(f"Componente {i+1}: {var:.2%}")
            print(f"\nVarianza explicada acumulada: {sum(self.var_explicada):.2%}")
            
            # Mostrar información adicional
            print("\nPrimeras componentes principales:")
            print(self.coordenadas_ind.head())
            
            print("\nCorrelaciones con las variables:")
            print(self.correlacion_var.round(3))
            
            return self.coordenadas_ind
            
        except Exception as e:
            print(f"Error durante el ACP: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def plot_plano_principal(self, ejes=[0, 1], ind_labels=None, titulo='Plano Principal'):
        """
        Grafica el plano principal del análisis de componentes principales.
        """
        if not hasattr(self, 'coordenadas_ind'):
            print("Debe ejecutar ACP primero.")
            return

        try:
            # Obtener coordenadas
            x = self.coordenadas_ind.iloc[:, ejes[0]]
            y = self.coordenadas_ind.iloc[:, ejes[1]]
            
            # Crear gráfico
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(x, y, c=range(len(x)), cmap='viridis', alpha=0.6)
            
            # Añadir etiquetas si se especifican
            if ind_labels is not None:
                for i, (xi, yi) in enumerate(zip(x, y)):
                    if i % 5 == 0:  # Etiquetar cada 5 puntos para evitar sobrecarga
                        plt.annotate(ind_labels[i], (xi, yi), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8)
            
            # Añadir líneas de referencia
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
            
            # Configurar etiquetas y título
            plt.xlabel(f'Componente {ejes[0]+1} ({self.var_explicada[ejes[0]]:.2%})')
            plt.ylabel(f'Componente {ejes[1]+1} ({self.var_explicada[ejes[1]]:.2%})')
            plt.title(titulo)
            plt.colorbar(scatter, label='Índice de observación')
            
            # Ajustar layout y mostrar
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error al graficar el plano principal: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def plot_circulo(self, ejes=[0, 1], var_labels=True, titulo='Círculo de Correlación'):
        """
        Grafica el círculo de correlación de las variables.
        """
        if not hasattr(self, 'correlacion_var'):
            print("Debe ejecutar ACP primero.")
            return
        
        try:
            # Obtener coordenadas de las variables
            cor = self.correlacion_var.iloc[:, ejes]
            
            # Crear gráfico con tamaño y márgenes ajustados
            plt.figure(figsize=(12, 10))
            
            # Dibujar círculo unitario
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='gray', alpha=0.5)
            plt.gca().add_patch(circle)
            
            # Configurar aspecto
            plt.axis('equal')  # Mantener proporción 1:1
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            
            # Establecer límites del gráfico
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)
            
            # Graficar variables con etiquetas mejoradas
            for i, (idx, row) in enumerate(cor.iterrows()):
                # Verificar que las coordenadas no sean nulas
                if not np.isnan(row[0]) and not np.isnan(row[1]):
                    # Dibujar flecha
                    plt.arrow(0, 0, row[0], row[1], 
                            color='steelblue', alpha=0.7,
                            head_width=0.03, head_length=0.03,
                            length_includes_head=True)
                    
                    if var_labels:
                        # Acortar nombres de variables si son muy largos
                        label = idx if len(idx) < 20 else idx[:17] + '...'
                        
                        # Calcular posición de la etiqueta
                        label_x = row[0] * 1.1
                        label_y = row[1] * 1.1
                        
                        # Añadir etiqueta con fondo blanco para mejor visibilidad
                        plt.annotate(label, 
                                   (label_x, label_y),
                                   color='darkblue',
                                   ha='center', va='center',
                                   fontsize=8,
                                   bbox=dict(facecolor='white', 
                                           edgecolor='none',
                                           alpha=0.8,
                                           pad=2))
            
            # Configurar etiquetas y título con tamaños ajustados
            plt.xlabel(f'Componente {ejes[0]+1} ({self.var_explicada[ejes[0]]:.2%})', fontsize=10)
            plt.ylabel(f'Componente {ejes[1]+1} ({self.var_explicada[ejes[1]]:.2%})', fontsize=10)
            plt.title(titulo, pad=20, fontsize=12)
            
            # Ajustar layout con márgenes específicos
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
            
            # Añadir una leyenda con la interpretación
            plt.figtext(0.99, 0.01, 
                       'Las flechas indican la correlación de las variables\ncon las componentes principales',
                       ha='right', va='bottom', fontsize=8, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.show()
            
        except Exception as e:
            print(f"Error al graficar el círculo de correlación: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def plot_sobreposicion(self, ejes=[0, 1], ind_labels=True, var_labels=True, titulo='Biplot: Variables e Individuos'):
        """
        Combina el gráfico del plano principal con el círculo de correlación.
        """
        if not hasattr(self, 'coordenadas_ind') or not hasattr(self, 'correlacion_var'):
            print("Debe ejecutar ACP primero.")
            return

        try:
            # Obtener coordenadas
            coord_ind = self.coordenadas_ind.iloc[:, ejes]
            coord_var = self.correlacion_var.iloc[:, ejes]
            
            # Calcular factor de escala para las variables
            scale_ind = np.max([coord_ind.iloc[:,0].abs().max(), coord_ind.iloc[:,1].abs().max()])
            scale_var = np.max([coord_var.iloc[:,0].abs().max(), coord_var.iloc[:,1].abs().max()])
            scale = (scale_ind / scale_var) * 0.7
            
            # Crear gráfico con márgenes ajustados
            plt.figure(figsize=(14, 10))
            
            # Graficar individuos
            scatter = plt.scatter(coord_ind.iloc[:,0], coord_ind.iloc[:,1], 
                                c=range(len(coord_ind)), cmap='viridis',
                                alpha=0.5, label='Observaciones')
            
            # Graficar variables con etiquetas más pequeñas
            for i, (idx, row) in enumerate(coord_var.iterrows()):
                plt.arrow(0, 0, row[0]*scale, row[1]*scale, 
                         color='red', alpha=0.5, head_width=0.05*scale)
                if var_labels:
                    # Acortar nombres de variables si son muy largos
                    label = idx if len(idx) < 20 else idx[:17] + '...'
                    plt.annotate(label, 
                               (row[0]*scale*1.1, row[1]*scale*1.1),
                               color='red', ha='center', va='center',
                               fontsize=8)  # Reducir tamaño de fuente
            
            # Configurar aspecto
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            plt.grid(True, alpha=0.3)
            
            # Configurar etiquetas y título con márgenes ajustados
            plt.xlabel(f'Componente {ejes[0]+1} ({self.var_explicada[ejes[0]]:.2%})', fontsize=10)
            plt.ylabel(f'Componente {ejes[1]+1} ({self.var_explicada[ejes[1]]:.2%})', fontsize=10)
            plt.title(titulo, pad=20, fontsize=12)
            
            # Añadir colorbar y leyenda con tamaño reducido
            cbar = plt.colorbar(scatter, label='Índice de observación')
            cbar.ax.tick_params(labelsize=8)
            plt.legend(['Observaciones', 'Variables'], fontsize=8)
            
            # Ajustar layout con márgenes específicos
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
            plt.show()
            
        except Exception as e:
            print(f"Error al graficar la sobreposición: {str(e)}")
            import traceback
            print(traceback.format_exc())
    

    def KMEDIAS(self, n_clusters=3, metodo='kmeans'):
        """
        Realiza clustering usando K-means o K-medoids.
        """
        if not self.verificar_datos():
            return None
            
        try:
            # Seleccionar el modelo
            if metodo == 'kmeans':
                modelo = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            elif metodo == 'kmedoids':
                if not globals().get('SKLEARN_EXTRA_AVAILABLE', False):
                    print("Error: scikit-learn-extra no está instalado. No se puede usar K-medoids.")
                    return None
                modelo = KMedoids(n_clusters=n_clusters, random_state=42)
            else:
                print(f"Método {metodo} no reconocido. Usando 'kmeans' por defecto.")
                modelo = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                
            # Ajustar el modelo
            clusters = modelo.fit_predict(self.df_normalizada)
            
            # Reducción de dimensionalidad para visualización
            pca = PCA(n_components=2)
            coords = pca.fit_transform(self.df_normalizada)
            
            # Visualizar clusters
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap='viridis')
            
            # Plotear centroides
            if hasattr(modelo, 'cluster_centers_'):
                centroids = pca.transform(modelo.cluster_centers_)
                plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroides')
                
            plt.title(f'Clusters usando {metodo.upper()}')
            plt.xlabel('Primera Componente Principal')
            plt.ylabel('Segunda Componente Principal')
            plt.colorbar(scatter, label='Cluster')
            plt.legend()
            plt.show()
            
            # Evaluar la calidad del clustering
            if hasattr(modelo, 'inertia_'):
                print(f"Inercia del clustering: {modelo.inertia_:.2f}")
                
            return clusters
            
        except Exception as e:
            print(f"Error durante el clustering: {str(e)}")
            return None

    def TSNE(self, n_componentes):
        df_numerico = self.analisisNumerico()
        if df_numerico is None:
            print("El DataFrame seleccionado en 'analisisNumerico()' está vacío o no existe.")
            return
        componentes = TSNE(n_components=n_componentes).fit_transform(df_numerico)
        self._plot_scatter(componentes, 'T-SNE')
            

    def UMAP(self, n_componentes, n_neighbors):
        """
        Realiza reducción de dimensionalidad usando UMAP.
        """
        if not globals().get('UMAP_AVAILABLE', False):
            print("Error: umap-learn no está instalado. No se puede usar UMAP.")
            return None
            
        try:
            df_numerico = self.analisisNumerico()
            if df_numerico is None:
                print("El DataFrame seleccionado en 'analisisNumerico()' está vacío o no existe.")
                return
                
            componentes = um.UMAP(n_components=n_componentes, n_neighbors=n_neighbors).fit_transform(df_numerico)
            self._plot_scatter(componentes, 'UMAP')
        except Exception as e:
            print(f"Error durante UMAP: {str(e)}")
            return None

    def _plot_scatter(self, componentes, title):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        ax.scatter(componentes[:, 0], componentes[:, 1])
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_title(title)
        ax.grid(False)
        plt.show()

from IPython.display import display
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

class AnalisisDatosBase:
    def __init__(self, file_path, delimiter=',', decimal='.'):
        """Clase base que maneja la carga de datos desde archivos"""
        self.file_path = file_path
        self.delimiter = delimiter
        self.decimal = decimal
        self.df = self._cargar_datos()
        
    def _cargar_datos(self):
        """Carga los datos desde el archivo CSV"""
        try:
            df = pd.read_csv(self.file_path, delimiter=self.delimiter, decimal=self.decimal)
            df = df.dropna()
            print(f"Datos cargados correctamente. Dimensiones: {df.shape}")
            return df
        except Exception as e:
            print(f"Error al cargar datos: {str(e)}")
            return None
    
    def resumen_general(self):
        """Muestra información básica del dataset"""
        if self.df is None:
            print("No hay datos cargados")
            return
            
        print("\nInformación del dataset:")
        print(self.df.info())
        
        print("\nEstadísticas descriptivas:")
        display(self.df.describe())
        
    def graficar_relaciones(self):
        """Muestra gráficos de relaciones entre variables"""
        if self.df is None:
            print("No hay datos cargados")
            return
            
        numeric_data = self.df.select_dtypes(include=['number'])
        if len(numeric_data.columns) < 2:
            print("No hay suficientes columnas numéricas para graficar")
            return
            
        # Mapa de calor de correlaciones
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Mapa de Correlación')
        plt.show()

class Supervisado(AnalisisDatosBase):
    def __init__(self, file_path, target_column, delimiter=',', decimal='.'):
        """
        Clase para análisis supervisado
        
        Args:
            file_path: Ruta al archivo CSV
            target_column: Nombre de la columna objetivo
            delimiter: Delimitador del CSV
            decimal: Separador decimal
        """
        super().__init__(file_path, delimiter, decimal)
        self.target_column = target_column
        
        if self.df is not None and target_column not in self.df.columns:
            print(f"Error: La columna '{target_column}' no existe en los datos")
            self.df = None

    @staticmethod
    def calcular_metricas(y_real, y_pred):
        """Calcula métricas de evaluación"""
        return {
            "R2": r2_score(y_real, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
            "MAE": np.mean(np.abs(y_real - y_pred)),
            "ER": (np.mean(np.abs(y_real - y_pred)) / np.mean(y_real) * 100)
        }
    
    def _preparar_datos(self, feature_columns=None):
        """Prepara los datos para modelado"""
        if self.df is None:
            return None, None, None, None
            
        # Si no se especifican features, usar todas las numéricas excepto target
        if feature_columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            feature_columns = [col for col in numeric_cols if col != self.target_column]
        
        X = self.df[feature_columns]
        y = self.df[self.target_column]
        
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def regresion_simple(self, feature_column=None):
        """Regresión lineal simple"""
        if feature_column is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            feature_column = [col for col in numeric_cols if col != self.target_column][0]
            
        X_train, X_test, y_train, y_test = self._preparar_datos([feature_column])
        
        modelo = LinearRegression().fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        print(f"Coeficiente: {modelo.coef_[0]:.4f}")
        print(f"Intercepto: {modelo.intercept_:.4f}")
        
        return self.calcular_metricas(y_test, y_pred)

    def comparar_modelos(self):
        """Compara múltiples algoritmos de regresión"""
        if self.df is None:
            return None
            
        resultados = []
        X_train, X_test, y_train, y_test = self._preparar_datos()
        
        # Modelos a comparar
        modelos = {
            "Regresión Lineal": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "Árbol de Decisión": DecisionTreeRegressor(max_depth=5),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor()
        }
        
        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            metricas = self.calcular_metricas(y_test, y_pred)
            resultados.append({"Modelo": nombre, **metricas})
        
        # Crear y mostrar resultados
        df_resultados = pd.DataFrame(resultados)
        display(df_resultados)
        
        # Gráfico comparativo
        df_resultados.set_index("Modelo").plot(kind='bar', figsize=(10, 6), 
                                             title="Comparación de Modelos")
        plt.ylabel("Valor de la Métrica")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return df_resultados

class AnalisisDatosClasificacion:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def cargar_datos(self):        
        self.data = pd.read_csv(self.file_path)
        self.data = self.data.dropna()  # Elimina valores nulos        

    def dataVisualization(self):        
        num_columns = len(self.data.columns)
        columns_per_plot = 10  # Ajusta según tu preferencia

        

        # Graficar mapa de calor de correlaciones
        correlation = self.data.corr()
        plt.figure(figsize=(18, 15))  # Aumentar tamaño
        plt.title('Correlation of Attributes with Class variable')

        # Agregar anotaciones (valores) en cada casilla
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

        plt.xticks(rotation=45, ha='right')  # Rotar y alinear etiquetas del eje X
        plt.yticks(rotation=0)
        plt.tight_layout()  # Ajustar espaciado
        plt.show()
        
    def preprocess_data(self):
        # Definir X (características) y y (target)
        X = self.data.drop('Bankrupt?', axis=1)  # Eliminar la columna 'Bankrupt?' de las características
        y = self.data['Bankrupt?']

        # Dividir los datos en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Verifica si la división fue correcta

        cols = self.X_train.columns
        scaler = StandardScaler()
        
        # Escalar las características
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        # Asignar los datos escalados a los atributos
        self.X_train = pd.DataFrame(X_train_scaled, columns=cols)
        self.X_test = pd.DataFrame(X_test_scaled, columns=cols)
        

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def EDA(self):
        print("EDA del dataset\n")
        print(f"Shape de los datos: {self.data.shape}\n")
        print(f"Primeras 10 filas de los datos: {self.data.head()}\n")
        print("Información del dataset:")
        self.data.info()
        print("\n")
        print(f"Estadisticas del dataset:{round(self.data.describe(), 2)}")
        self.dataVisualization()

class evaluador(AnalisisDatosClasificacion):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.cargar_datos()
        self.preprocess_data()

    def comparar_algoritmos(self):
        # Diccionario de modelos a evaluar
        modelos = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(random_state=0, max_iter=1000),
            "Decision Tree 3": DecisionTreeClassifier(
                max_depth=12, 
                min_samples_split=5, 
                class_weight='balanced', 
                criterion='gini', 
                random_state=0
            ),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0),
            "SVM": SVC(kernel='linear', probability=True, random_state=0)
        }

        resultados = []  # Para almacenar resultados de cada modelo

        for nombre, modelo in modelos.items():
            print(f"Evaluando modelo: {nombre}\n")

            # Entrenar el modelo
            modelo.fit(self.X_train, self.y_train)

            # Predicciones
            y_pred = modelo.predict(self.X_test)
            y_pred_prob = modelo.predict_proba(self.X_test)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(self.X_test)

            # Calcular métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_prob)
            report = classification_report(self.y_test, y_pred, output_dict=True)

            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']

            cm = confusion_matrix(self.y_test, y_pred)

            # Mostrar matriz de confusión
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
            plt.title(f"Matriz de Confusión - {nombre}")
            plt.ylabel('Clase Verdadera')
            plt.xlabel('Clase Predicha')
            plt.show()

            print(f"Accuracy: {accuracy}")
            print(f"AUC: {auc}\n")
            print(f"Classification Report:\n{classification_report(self.y_test, y_pred)}\n")

            # Guardar resultados
            resultados.append({
                "Modelo": nombre, 
                "Accuracy": accuracy, 
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score
            })

        # Crear un DataFrame con los resultados
        resultados_df = pd.DataFrame(resultados)
        print("\nResumen de Resultados:")
        print(resultados_df)

        # Visualizar resultados en un gráfico
        resultados_df.set_index("Modelo")[["Accuracy", "AUC", "Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(10, 6), rot=45)
        plt.title("Comparación de Algoritmos")
        plt.ylabel("Puntuación")
        plt.legend(loc="lower right")
        plt.show()

        # Crear una tabla con los resultados usando Matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        table_data = resultados_df.values
        column_labels = resultados_df.columns
        table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(column_labels))))

        plt.title("Resumen de Resultados")
        plt.show()


class SerieTiempo:
    def __init__(self, columna_fecha='Fecha', columna_valor='Valor', delimiter_type=1):
        # Inicializamos sin path ni DataFrame
        self.__path = None
        self.__columna_fecha = columna_fecha
        self.__columna_valor = columna_valor
        self.__delimiter_type = delimiter_type
        self.__df = None
        self.__frecuencia = None
        self.train_data = None
        self.test_data = None

    # Método privado para cargar el dataset de series de tiempo
    def __cargar_datos(self):
        # Carga el DataFrame según el delimitador especificado
        try:
            if self.__delimiter_type == 1:
                df = pd.read_csv(self.__path, sep=",", decimal=".", encoding='utf-8')
            elif self.__delimiter_type == 2:
                df = pd.read_csv(self.__path, sep=";", decimal=",", encoding='utf-8')
            else:
                raise ValueError("Unsupported delimiter type")
        except UnicodeDecodeError:
            # Intentar con latin1 si falla la codificación utf-8
            if self.__delimiter_type == 1:
                df = pd.read_csv(self.__path, sep=",", decimal=".", encoding='latin1')
            elif self.__delimiter_type == 2:
                df = pd.read_csv(self.__path, sep=";", decimal=",", encoding='latin1')
        
        # Convertir columnas a los tipos correctos
        df[self.__columna_valor] = df[self.__columna_valor].astype(float)
        df[self.__columna_fecha] = pd.to_datetime(df[self.__columna_fecha], errors='coerce')
        
        return df
        
    # Propiedad para el path
    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, new_path):
        # Asigna una nueva ruta y recarga el dataset
        self.__path = new_path
        self.__df = self.__cargar_datos()

    # Propiedad para acceder al DataFrame cargado
    @property
    def df(self):
        # Verifica que el DataFrame esté cargado antes de devolverlo
        if self.__df is None:
            raise ValueError("El dataset aún no se ha cargado. Asigna una ruta primero con `path`.")
        return self.__df

    # Métodos para inspeccionar el DataFrame
    def mostrar_tipos_datos(self):
        return self.__df.dtypes

    def mostrar_valores_nulos(self):
        return self.__df.isna().sum()

    def mostrar_head(self, n=5):
        return self.__df.head(n)
    
    def eliminar_fechas_invalidas(self):
        # Eliminar filas con fechas no válidas
        self.__df = self.__df.dropna(subset=[self.__columna_fecha])

    def generar_rango_fechas(self, fecha_inicio, fecha_final, frecuencia="10min"):
        # Guardar la frecuencia como atributo para que otras funciones la usen
        self.__frecuencia = frecuencia
        
        # Crear el rango completo de fechas
        total_fechas = pd.date_range(start=fecha_inicio, end=fecha_final, freq=frecuencia).tolist()
        fechas_actuales = pd.DatetimeIndex(self.__df[self.__columna_fecha].values)
        self.__fechas_faltantes = [x for x in total_fechas if x not in fechas_actuales]

    def verificar_fechas_faltantes(self):
        # Verifica si se han generado fechas faltantes y las retorna
        if self.__fechas_faltantes is None:
            raise ValueError("Debe generar el rango de fechas primero con `generar_rango_fechas()`.")
        return self.__fechas_faltantes

    def mostrar_fechas_faltantes(self):
        # Muestra las fechas faltantes en un DataFrame
        fechas_faltantes = self.verificar_fechas_faltantes()
        df_fechas_faltantes = pd.DataFrame({self.__columna_fecha: fechas_faltantes})
    
        # Unir con el DataFrame original para agregar la columna de valor en las fechas faltantes
        df_fechas_faltantes = pd.merge(df_fechas_faltantes, self.__df[[self.__columna_fecha, self.__columna_valor]], 
                                       on=self.__columna_fecha, how='left')
        return df_fechas_faltantes
    
    def imputar_fechas_faltantes(self):
        # Imputar valores en fechas faltantes usando media móvil
        fechas_faltantes = self.verificar_fechas_faltantes()
        
        if not self.__fechas_faltantes:
            print("No hay fechas faltantes para imputar.")
            return
    
        # Crear una copia del DataFrame antes de modificarlo
        df_copia = self.__df.copy()
    
        # Concatenar las fechas faltantes a la copia del DataFrame
        df_fechas_faltantes = pd.DataFrame({self.__columna_fecha: self.__fechas_faltantes})
        df_copia = pd.concat([df_copia, df_fechas_faltantes], ignore_index=True)
        df_copia = df_copia.sort_values(by=[self.__columna_fecha])
    
        # Aplicar suavizado con media móvil en los valores
        df_copia[self.__columna_valor] = df_copia[self.__columna_valor].fillna(
            df_copia[self.__columna_valor].rolling(5, min_periods=1, center=True).mean()
        )
    
        print("Fechas faltantes imputadas y suavizadas correctamente.")
        return df_copia

    def crear_serie_temporal(self, df=None, graficar=False, plotly_plot=False):
        # Si no se pasa un DataFrame, usar el DataFrame original
        if df is None:
            df = self.__df
    
        fechas = pd.DatetimeIndex(df[self.__columna_fecha])
        ts = pd.Series(df[self.__columna_valor].values, index=fechas)
        ts_df = ts.to_frame(name="valor")
        
        # Si el parámetro graficar es True y se solicita un gráfico con plotly
        if graficar:
            if plotly_plot:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_df.index.tolist(), y=ts_df['valor'].tolist(), mode='lines+markers'))
                # Agregar el slider en el eje X
                fig.update_xaxes(rangeslider_visible=True)
                fig.show()
            else:
                # Plot con matplotlib
                ts.plot(title='Serie Temporal')
                plt.show()
        
        return ts

    def dividir_datos(self, periodos_prediccion, df=None):
        # Usar el DataFrame proporcionado o el cargado
        if df is None:
            df = self.__df
        
        # Reindexar con la frecuencia especificada
        df = df.set_index(self.__columna_fecha).asfreq(self.__frecuencia)
        ts = df[self.__columna_valor]
        
        # Dividir en datos de entrenamiento y prueba
        self.train_data = ts[:-periodos_prediccion]
        self.test_data = ts[-periodos_prediccion:]
    
    def ejecutar_todos_los_modelos(self, periodos_prediccion, df=None):
        # Dividir datos si se especifica un DataFrame o usar el cargado
        self.dividir_datos(periodos_prediccion, df)

        resultados = {
            'holt_winters': self._predecir_holt_winters(periodos_prediccion),
            'holt_winters_calibrado': self._predecir_holt_winters_calibrado(periodos_prediccion),
            'arima': self._predecir_arima(periodos_prediccion),
            'arima_calibrado': self._predecir_arima_calibrado(periodos_prediccion),
            'lstm': self._predecir_lstm(periodos_prediccion)
        }
        return resultados

    def _predecir_holt_winters(self, periodos_prediccion):
        frecuencia_ciclos = {
            'D': 365, 'W': 52, 'M': 12, 'Q': 4, 'H': 24
        }
        seasonal_periods = frecuencia_ciclos.get(self.__frecuencia[0].upper(), None)
        if seasonal_periods is None:
            raise ValueError(f"No se puede determinar el ciclo estacional para la frecuencia '{self.__frecuencia}'")

        modelo = ExponentialSmoothing(self.train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
        modelo_fit = modelo.fit()
        predicciones = modelo_fit.forecast(periodos_prediccion)
        
        return self.train_data, self.test_data, predicciones

    def _predecir_holt_winters_calibrado(self, periodos_prediccion):
        seasonal_periods = 12  # Ajusta esto según tus datos
        modelo = ExponentialSmoothing(
            self.train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods
        ).fit()
        predicciones = modelo.forecast(periodos_prediccion)
        
        return self.train_data, self.test_data, predicciones

    def _predecir_arima(self, periodos_prediccion, order=(1, 1, 1)):
        modelo = ARIMA(self.train_data, order=order)
        modelo_fit = modelo.fit()
        predicciones = modelo_fit.forecast(steps=periodos_prediccion)

        return self.train_data, self.test_data, predicciones

    def _predecir_arima_calibrado(self, periodos_prediccion):
        # Utiliza auto_arima para encontrar los mejores parámetros
        auto_model = auto_arima(self.train_data, seasonal=False, stepwise=True, error_action='ignore', suppress_warnings=True)
        modelo_fit = auto_model.fit(self.train_data)
        predicciones = modelo_fit.predict(n_periods=periodos_prediccion)

        return self.train_data, self.test_data, pd.Series(predicciones, index=self.test_data.index)

    def _predecir_lstm(self, periodos_prediccion, n_steps=10):
        ts = self.train_data.values
        X, y = [], []
        for i in range(len(ts) - n_steps):
            X.append(ts[i:i+n_steps])
            y.append(ts[i+n_steps])
        X, y = np.array(X), np.array(y)
        
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape para LSTM
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=200, verbose=0)

        input_seq = ts[-n_steps:]
        predicciones = []
        for _ in range(periodos_prediccion):
            input_seq = input_seq.reshape((1, n_steps, 1))
            pred = model.predict(input_seq, verbose=0)
            predicciones.append(pred[0, 0])
            input_seq = np.append(input_seq[0, 1:], pred)

        return self.train_data, self.test_data, np.array(predicciones)

    # Método para graficar múltiples modelos
    def graficar_predicciones_multiples(self, resultados, agregar_slider=False):
        fig = go.Figure()
    
        # Graficar datos de entrenamiento y prueba (comunes para todos los modelos)
        train_data, test_data = list(resultados.values())[0][:2]  # Obtener train_data y test_data de cualquier modelo
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data.values, mode='lines+markers', name='Entrenamiento'))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values, mode='lines+markers', name='Prueba'))
    
        # Graficar las predicciones de cada modelo
        for modelo, (train_data, test_data, predicciones) in resultados.items():
            # Convertir `predicciones` en `pd.Series` para usar las fechas de `test_data` como índice
            predicciones = pd.Series(predicciones, index=test_data.index)
            fig.add_trace(go.Scatter(x=predicciones.index, y=predicciones.values, mode='lines+markers', name=modelo))
    
        # Agregar slider si se especifica
        if agregar_slider:
            fig.update_xaxes(rangeslider_visible=True)
    
        fig.show()
    
    # Método para calcular métricas
    def calcular_metricas(self, resultados):
        # Lista para almacenar los resultados de cada modelo
        metricas = []
    
        # Calcular métricas para cada modelo
        for modelo, (train_data, test_data, predicciones) in resultados.items():
            # Convertir predicciones y test_data a Series si es necesario
            predicciones = pd.Series(predicciones, index=test_data.index)
    
            # Calcular MSE y RMSE
            mse = mean_squared_error(test_data, predicciones)
            rmse = np.sqrt(mse)
    
            # Calcular Error Relativo (RE)
            re = mean_absolute_error(test_data, predicciones) / test_data.mean()
    
            # Calcular correlación
            corr = test_data.corr(predicciones)
    
            # Agregar resultados a la lista
            metricas.append({
                'Modelo': modelo,
                'MSE': mse,
                'RMSE': rmse,
                'RE': re,
                'CORR': corr
            })
    
        # Crear DataFrame con las métricas
        df_metricas = pd.DataFrame(metricas)
        df_metricas.set_index('Modelo', inplace=True)
    
        return df_metricas
