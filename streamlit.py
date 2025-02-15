import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Análisis de valor de Agua",
    page_icon="💧",
    layout="wide"
)

# Title and description
st.title("📊 Dashboard de valor de Agua")
st.markdown("Análisis y visualización de datos de valor de agua")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('consumo_agua.csv', sep=';')
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

df = load_data()

if df is not None:
    # Convert fecha to datetime first
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Add sidebar
    st.sidebar.header("Filtros")
    
    # Add year filter
    df['año'] = df['fecha'].dt.year
    años = sorted(df['año'].unique())
    año_seleccionado = st.sidebar.selectbox(
        "Seleccionar Año",
        options=años,
        index=len(años)-1
    )
    
    # Create mes column
    df['mes'] = df['fecha'].dt.month
    
    # Add month filter
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    mes_seleccionado = st.sidebar.selectbox(
        "Seleccionar Mes",
        options=list(meses.keys()),
        format_func=lambda x: meses[x],
        index=0
    )
    
    # Filter data based on selection
    df_filtered = df[
        (df['año'] == año_seleccionado) &
        (df['mes'] == mes_seleccionado)
    ]
    
    # Add summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resumen del período seleccionado")
    st.sidebar.metric(
        label="Valor total",
        value=f"${df_filtered['valor'].sum():,.2f}"
    )
    st.sidebar.metric(
        label="Promedio mensual",
        value=f"${df_filtered['valor'].mean():,.2f}"
    )

    # Show basic information about the dataset
    st.subheader("Vista previa de los datos filtrados")
    st.dataframe(df_filtered)
    
    # Basic statistics
    st.subheader("Estadísticas básicas")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Valor promedio",
            value=f"${df['valor'].mean():,.2f}"
        )
    
    with col2:
        st.metric(
            label="Valor máximo",
            value=f"${df['valor'].max():,.2f}"
        )
    
    with col3:
        st.metric(
            label="Valor mínimo",
            value=f"${df['valor'].min():,.2f}"
        )
    
    # Time series plot
    st.subheader("Tendencia de valor a lo largo del tiempo")
    fig = px.line(
        df,
        x='fecha',
        y='valor',
        title='Valor de agua a lo largo del tiempo'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly average consumption
    st.subheader("Valor promedio mensual")
    monthly_avg = df.groupby('mes')['valor'].mean().reset_index()
    
    fig_monthly = px.bar(
        monthly_avg,
        x='mes',
        y='valor',
        title='Valor promedio por mes',
        labels={'mes': 'Mes', 'valor': 'Valor promedio ($)'}
    )
    
    # Update x-axis to show month names
    fig_monthly.update_xaxes(
        ticktext=list(meses.values()),
        tickvals=list(meses.keys())
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Download data option
    st.subheader("Descargar datos")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name="Valor_agua_processed.csv",
        mime="text/csv"
    )
else:
    st.warning("Por favor, asegúrate de que el archivo 'consumo_agua.csv' existe en el directorio del proyecto.")
