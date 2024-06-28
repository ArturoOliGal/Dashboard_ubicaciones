import streamlit as st
import pandas as pd
import seaborn as sbn
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px

st.title('Dashboard ubicaciones')
st.markdown('Pocisiones incorrectas, avance de ubicaciones totales y detallado')
st.sidebar.header('Filtro de dashboard')

#https://docs.google.com/spreadsheets/d/e/2PACX-1vRzkqXVc1RhH_NbP0LrXF8G6JHMMuSopxWhWQareqsOtKR7lWsJgzfSfKkY1sZ130LsCR3YDqwbMCGU/pubhtml

@st.cache_data
def load_data():
    url='https://docs.google.com/spreadsheets/d/e/2PACX-1vRzkqXVc1RhH_NbP0LrXF8G6JHMMuSopxWhWQareqsOtKR7lWsJgzfSfKkY1sZ130LsCR3YDqwbMCGU/pubhtml'
    html=pd.read_html(url, header=1)
    df=html[0]
    df=df.dropna(subset=['Articulo'])
    df = df[['Articulo','Nombre producto','Presentacion','VP.Tot.May','Desc','Existencia total','Ubicacion de articulo','Fecha','VoF Etiquetas','Posiciones con etiquetas','VoF']]
    return df

def load_dataSKUS():
    url='https://docs.google.com/spreadsheets/d/e/2PACX-1vRzkqXVc1RhH_NbP0LrXF8G6JHMMuSopxWhWQareqsOtKR7lWsJgzfSfKkY1sZ130LsCR3YDqwbMCGU/pubhtml'
    html=pd.read_html(url, header=1)
    SKU=html[1]
    SKU = SKU[['Almacen','SKUs']]
    return SKU

df=load_data()
DB_SKU=load_dataSKUS()
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d-%b-%Y')

Fecha_min=df['Fecha'].min().date()
Fecha_max=df['Fecha'].max().date()
Start_date = st.sidebar.date_input('Fecha de inicio', value=Fecha_min, min_value=Fecha_min, max_value=Fecha_max)
End_date = st.sidebar.date_input('Fecha de fin', value=Fecha_max, min_value=Fecha_min, max_value=Fecha_max)
filtro_mapa = st.sidebar.selectbox('Filtro para mapa del almacen', ['Todos', 'CAJA','DET','JAUL','BULT','TPCO'])
filtered_DB = df[(df['Fecha'] >= pd.to_datetime(Start_date)) & (df['Fecha'] <= pd.to_datetime(End_date))]
filtro_por_articulo=st.sidebar.multiselect('Filtro de articulo', filtered_DB['Articulo'].unique())
filtro_por_nombre_procto=st.sidebar.multiselect('Filtro por producto', filtered_DB['Nombre producto'].unique())
mapa = pd.DataFrame(filtered_DB[['Ubicacion de articulo', 'Articulo','Nombre producto']])
#mapa['Articulo']=filtered_DB['Articulo']
mapa[['Almacen', 'Frente', 'Posicion','Nivel','Fondo']] = mapa['Ubicacion de articulo'].str.split('-', expand=True)
filtered_DB[['Almacen', 'Frente', 'Posicion','Nivel','Fondo']] = df['Ubicacion de articulo'].str.split('-', expand=True)


if filtro_mapa != 'Todos':
    mapa = mapa[mapa['Almacen'] == filtro_mapa]
    filtered_DB=filtered_DB[filtered_DB['Almacen']==filtro_mapa]


if filtro_por_nombre_procto:
    mapa = mapa[mapa['Nombre producto'].isin(filtro_por_nombre_procto)]
    filtered_DB = filtered_DB[filtered_DB['Nombre producto'].isin(filtro_por_nombre_procto)]


if filtro_por_articulo:
    mapa = mapa[mapa['Articulo'].isin(filtro_por_articulo)]
    filtered_DB = filtered_DB[filtered_DB['Articulo'].isin(filtro_por_articulo)]

#mapa=filtered_DB['Ubicacion de articulo']

mapa=mapa.dropna()
n = 1
mapa[['numero_frente', 'letra_frente']] = mapa['Frente'].str.extract(r'(\d+)([A-Za-z]+)', expand=True)
mapa=mapa[['Almacen','numero_frente','letra_frente','Posicion','Nivel','Fondo','Articulo']]

multiplier_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25}

mapa['Numero_frente'] = mapa['letra_frente'].apply( lambda x: multiplier_dict[x] * 1 if pd.notna(x) else 8)

#mapa['Coor_X']=mapa['Numero_frente']*mapa['Numero_almacen']
mapa['Coor_X']=mapa['Numero_frente']
mapa = mapa.dropna(subset=['Coor_X', 'Posicion'])

fig = px.scatter(mapa, x='Coor_X', y='Posicion', title='Mapa del almacen')
fig.update_layout(
    xaxis=dict(title=None, showticklabels=False),
    yaxis=dict(title=None, showticklabels=False)
)


Grupos = filtered_DB.groupby('Almacen')['VoF'].sum()
Grupos_fuera=filtered_DB.groupby('Almacen')['VoF Etiquetas'].sum()
#Grupos

col1, col2, col3 = st.columns(3)
with col1:
    st.header('Productos dados de alta')
    Grupos
    

with col2:
    st.header('Porcentaje dado de alta')
    posiciones_bien=filtered_DB['VoF'].sum()
    posiciones=df['Articulo'].count()
    Porcentaje_posiciones=round(((posiciones_bien/posiciones)*100),2)
    maximo=100
    color_gauge = "#522d6d"
    color_gray = "#e5e1e6"
    color_threshold = "red"
    gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=Porcentaje_posiciones,
            title={'text': "Indicador de Medidor"},
            gauge={
                'axis': {'range': [0, maximo]},
                'bar': {'color': color_gauge},
                'steps': [
                    {'range': [0, 50], 'color': color_gray},
                    {'range': [50, 100], 'color': color_gray}],
                'threshold': {
                    'line': {'color': color_gray, 'width': 4},
                    'thickness': 0.75,
                    'value': 100}}))
    st.plotly_chart(gauge)
    #st.markdown('Numero de productos dados de alta')
    #st.markdown(posiciones_bien)


with col3:
    st.header('Productos fuera de su lugar')
    Grupos_fuera

st.plotly_chart(fig)
filtered_DB
#df
#mapa
