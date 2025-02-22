# Dashboard de Análisis de Valor del Agua

## Descripción

Una aplicación web desarrollada con Streamlit para analizar y visualizar datos históricos del valor del agua. Esta herramienta permite a los usuarios explorar tendencias, patrones estacionales y estadísticas detalladas del consumo y valor del agua.

## Características

- 📊 Análisis numérico detallado
- 📈 Visualizaciones interactivas
- 🔍 Filtros dinámicos por:
  - Rango de fechas
  - Rango de valores
  - Año
  - Mes
  - Estación

## Visualizaciones Incluidas

- Tendencias temporales
- Promedios mensuales
- Comparación año a año
- Análisis estacional
- Distribución de valores
- Estadísticas básicas y avanzadas

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/tu-usuario/dashboard-valor-agua.git
cd dashboard-valor-agua
```


2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```


3. Ejecutar la aplicación:

```bash
streamlit run streamlit.py
```


## Requisitos

- Python 3.8 o superior
- Streamlit 1.28.0 o superior
- Pandas 2.0.0 o superior
- Plotly 5.18.0 o superior

## Estructura de Datos

El archivo `consumo_agua.csv` debe contener las siguientes columnas:
- `fecha`: Fecha del registro
- `valor`: Valor monetario del agua

## Uso

1. Inicie la aplicación con `streamlit run streamlit.py`
2. Utilice los filtros en la barra lateral para personalizar el análisis
3. Explore las diferentes visualizaciones y métricas
4. Descargue los datos filtrados en formato CSV si lo necesita

# TODO
1. Aplicar las visualizaciones y filtros para el modelo (split train/test)
2. Avanzar en las preguntas de investigación (un par de preguntas)