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
    
    # Create derived columns
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    
    # Create season mapping
    season_map = {
        12: 'Verano', 1: 'Verano', 2: 'Verano',
        3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
        6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
        9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
    }
    df['estacion'] = df['mes'].map(season_map)
    
    # Add sidebar
    st.sidebar.header("Filtros")
    
    # Add date range filter
    st.sidebar.subheader("Rango de Fechas")
    fecha_min = df['fecha'].min().date()
    fecha_max = df['fecha'].max().date()
    
    fecha_inicio = st.sidebar.date_input(
        "Fecha Inicial",
        value=fecha_min,
        min_value=fecha_min,
        max_value=fecha_max
    )
    
    fecha_fin = st.sidebar.date_input(
        "Fecha Final",
        value=fecha_max,
        min_value=fecha_min,
        max_value=fecha_max
    )
    
    # Add value range filter
    st.sidebar.subheader("Rango de Valores")
    valor_min, valor_max = st.sidebar.slider(
        "Seleccionar Rango de Valores ($)",
        float(df['valor'].min()),
        float(df['valor'].max()),
        (float(df['valor'].min()), float(df['valor'].max())),
        format="$%.2f"
    )
    
    # Add season filter
    st.sidebar.subheader("Estación")
    estacion_seleccionada = st.sidebar.multiselect(
        "Seleccionar Estaciones",
        options=['Verano', 'Otoño', 'Invierno', 'Primavera'],
        default=['Verano', 'Otoño', 'Invierno', 'Primavera']
    )
    
    # Add year filter
    años = sorted(df['año'].unique())
    año_seleccionado = st.sidebar.selectbox(
        "Seleccionar Año",
        options=años,
        index=len(años)-1
    )
    
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
    
    # Update filtered data with all filters
    df_filtered = df[
        (df['año'] == año_seleccionado) &
        (df['mes'] == mes_seleccionado) &
        (df['fecha'].dt.date >= fecha_inicio) &
        (df['fecha'].dt.date <= fecha_fin) &
        (df['valor'] >= valor_min) &
        (df['valor'] <= valor_max) &
        (df['estacion'].isin(estacion_seleccionada))
    ]
    
    # Add reset filters button
    if st.sidebar.button("Restablecer Filtros"):
        st.experimental_rerun()
    
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

    # Statistical Analysis Section
    st.header("📊 Análisis Numérico")
    
    # Show data preview first
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

    # Advanced statistics
    st.subheader("Análisis Estadístico Detallado")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Desviación Estándar", f"${df['valor'].std():,.2f}")
        st.metric("Mediana", f"${df['valor'].median():,.2f}")
    
    with col2:
        st.metric("Percentil 75", f"${df['valor'].quantile(0.75):,.2f}")
        st.metric("Percentil 25", f"${df['valor'].quantile(0.25):,.2f}")

    # Distribution histogram
    st.subheader("Distribución de Valores")
    num_bins = st.slider("Número de intervalos", min_value=10, max_value=50, value=20)
    
    fig_hist = px.histogram(
        df,
        x='valor',
        nbins=num_bins,
        title='Distribución de Valores',
        template='plotly_dark',
        color_discrete_sequence=['#00BFA5']
    )
    fig_hist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=20
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Visual Analysis Section
    st.header("📈 Análisis Visual")

    # Time series plot
    st.subheader("Tendencia de valor a lo largo del tiempo")
    fig = px.line(
        df,
        x='fecha',
        y='valor',
        title='Valor de agua a lo largo del tiempo',
        template='plotly_dark',
        color_discrete_sequence=['#00C9FF']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=20
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
        labels={'mes': 'Mes', 'valor': 'Valor promedio ($)'},
        template='plotly_dark',
        color_discrete_sequence=['#00E676']
    )
    
    fig_monthly.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=20
    )
    
    # Update x-axis to show month names
    fig_monthly.update_xaxes(
        ticktext=list(meses.values()),
        tickvals=list(meses.keys())
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Add year-over-year comparison
    st.subheader("Comparación Año a Año")
    yearly_avg = df.groupby('año')['valor'].agg(['mean', 'sum']).reset_index()
    col1, col2 = st.columns(2)
    
    with col1:
        fig_yearly = px.line(
            yearly_avg,
            x='año',
            y='mean',
            title='Promedio Anual',
            template='plotly_dark',
            color_discrete_sequence=['#FF4081']
        )
        fig_yearly.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20
        )
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    with col2:
        fig_yearly_total = px.bar(
            yearly_avg,
            x='año',
            y='sum',
            title='Total Anual',
            template='plotly_dark',
            color_discrete_sequence=['#7C4DFF']
        )
        fig_yearly_total.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20
        )
        st.plotly_chart(fig_yearly_total, use_container_width=True)

    # Add seasonal analysis
    st.subheader("Análisis Estacional")
    seasonal_avg = df.groupby('estacion')['valor'].mean().reset_index()
    
    fig_seasonal = px.pie(
        seasonal_avg,
        values='valor',
        names='estacion',
        title='Distribución Estacional del Consumo',
        template='plotly_dark',
        color_discrete_sequence=['#FF9800', '#4CAF50', '#2196F3', '#F44336']
    )
    fig_seasonal.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=20
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)

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
