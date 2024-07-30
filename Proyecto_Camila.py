import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def cargar_datos(archivo):
    """Carga los datos desde un archivo Excel en un DataFrame de Pandas."""
    try:
        df = pd.read_excel(archivo)
        st.write(f"Datos cargados con éxito.")
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return pd.DataFrame()

def transformar_datos(df):
    """Realiza transformaciones en los datos cargados."""
    try:
        # Ejemplo de transformación: eliminar filas con valores nulos en columnas importantes
        df = df.dropna(subset=['SafetySecurity', 'SocialCapital', 'InvestmentEnvironment', 'EconomicQuality'])
        
        # Convertir las columnas a tipo numérico si es necesario
        for col in ['SafetySecurity', 'SocialCapital', 'InvestmentEnvironment', 'EconomicQuality']:
            if df[col].dtype != 'float64' and df[col].dtype != 'int64':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error al transformar los datos: {e}")
        return pd.DataFrame()

def calcular_estadisticas(df):
    """Calcula estadísticas básicas para los indicadores."""
    try:
        # Suma total de unidades por indicador y país
        total_safety = df.groupby('Country')['SafetySecurity'].sum().sort_values(ascending=False)
        total_social = df.groupby('Country')['SocialCapital'].sum().sort_values(ascending=False)
        total_investment = df.groupby('Country')['InvestmentEnvironment'].sum().sort_values(ascending=False)
        total_economic = df.groupby('Country')['EconomicQuality'].sum().sort_values(ascending=False)

        return total_safety, total_social, total_investment, total_economic
    except Exception as e:
        st.error(f"Error al calcular las estadísticas: {e}")
        return None, None, None, None

def calcular_r2(x, y):
    """Calcula el valor de R^2 para la regresión lineal entre dos variables."""
    model = LinearRegression()
    x = x.values.reshape(-1, 1)  # Convertir a formato 2D requerido por LinearRegression
    model.fit(x, y)
    y_pred = model.predict(x)
    return r2_score(y, y_pred), model.intercept_, model.coef_[0]

def generar_graficos(df, total_safety, total_social, total_investment, total_economic):
    """Genera gráficos para la visualización de los datos con explicaciones previas."""
    try:
        # Explicación y gráfico de Box Plot para la distribución de indicadores
        st.write("## Distribución de Indicadores por País")
        st.write("""
        **Distribución de Indicadores por País:**

        Este gráfico muestra la distribución de diferentes indicadores para cada país en tu conjunto de datos.
        - **Eje X:** `variable`, que representa los diferentes indicadores (por ejemplo, Seguridad, Capital Social, Entorno de Inversión, Calidad Económica).
        - **Eje Y:** `value`, que muestra los valores de estos indicadores.

        **Componentes del Box Plot:**
        - **Caja (Box):** Muestra el rango intercuartílico (IQR) donde se encuentra el 50% central de los datos.
        - **Líneas (Whiskers):** Indican la extensión de los datos fuera del rango intercuartílico.
        - **Puntos (Outliers):** Valores atípicos que se encuentran fuera del rango de los bigotes.

        **Interpretación:** Ayuda a entender la variabilidad y la distribución de los indicadores. Identifica medianas, rangos y posibles valores atípicos entre los indicadores para los países.
        """)
        plt.figure(figsize=(12, 8))
        melted_df = df.melt(id_vars='Country', value_vars=['SafetySecurity', 'SocialCapital', 'InvestmentEnvironment', 'EconomicQuality'])
        sns.boxplot(x='variable', y='value', data=melted_df)
        plt.title('Distribución de Indicadores')
        plt.xlabel('Indicador')
        plt.ylabel('Valor')
        plt.xticks(rotation=45)  # Rotar etiquetas del eje X para mejor legibilidad
        plt.tight_layout()  # Ajustar diseño
        st.pyplot(plt.gcf())  # Mostrar el gráfico en Streamlit
        plt.clf()  # Limpiar la figura para evitar superposición con otros gráficos

        # Explicación y gráfico de Regresión Lineal entre Seguridad y Capital Social
        st.write("## Regresión Lineal entre Seguridad y Capital Social")
        st.write("""
        **Regresión Lineal entre Seguridad y Capital Social:**

        Este gráfico muestra la relación entre el indicador de `Seguridad` y el `Capital Social` usando una regresión lineal.
        - **Eje X:** `SafetySecurity`, que representa los puntajes de Seguridad para los países.
        - **Eje Y:** `SocialCapital`, que representa los puntajes de Capital Social para los países.

        **Componentes:**
        - **Puntos de Datos (Scatter Plot):** Representan la relación entre `Seguridad` y `Capital Social` para cada país.
        - **Línea de Regresión (Reg Plot):** Muestra la tendencia general en la relación entre las dos variables. La línea roja indica la mejor línea de ajuste para los datos.


        **Interpretación del Valor de \( R^2 \):**
        El valor de \( R^2 \) indica qué proporción de la variabilidad en el `Capital Social` se puede explicar por el `Seguridad`. Un valor de \( R^2 \) cercano a 1 significa que la línea de regresión explica bien la relación entre `Seguridad` y `Capital Social`, mientras que un valor cercano a 0 indica una relación débil.

        **Ecuación en el Gráfico:**
        En el gráfico, la ecuación de la línea de regresión y el valor de \( R^2 \) están indicados para proporcionar una comprensión visual de la relación entre `Seguridad` y `Capital Social`.
        """)
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='SafetySecurity', y='SocialCapital', data=df)
        plt.title('Regresión Lineal entre Seguridad y Capital Social')
        plt.xlabel('Seguridad')
        plt.ylabel('Capital Social')
        sns.regplot(x='SafetySecurity', y='SocialCapital', data=df, scatter=False, color='red')
        
        # Calcular y mostrar la ecuación
        r2, intercept, coef = calcular_r2(df['SafetySecurity'], df['SocialCapital'])
        ecuacion = f'$y = {intercept:.2f} + {coef:.2f}x$'
        st.write(f"**Ecuación de la Regresión Lineal:** {ecuacion}")
        st.write(f"**Valor de \( R^2 \):** {r2:.2f}")
        
        # Añadir la ecuación al gráfico
        plt.annotate(ecuacion, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red', backgroundcolor='white')
        plt.annotate(f'$R^2 = {r2:.2f}$', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='red', backgroundcolor='white')
        plt.tight_layout()  # Ajustar diseño
        st.pyplot(plt.gcf())  # Mostrar el gráfico en Streamlit
        plt.clf()  # Limpiar la figura para evitar superposición con otros gráficos

        # Explicación y gráfico de Regresión Lineal entre Entorno de Inversión y Calidad Económica
        st.write("## Regresión Lineal entre Entorno de Inversión y Calidad Económica")
        st.write("""
        **Regresión Lineal entre Entorno de Inversión y Calidad Económica:**

        Este gráfico muestra la relación entre el indicador de `Entorno de Inversión` y `Calidad Económica` usando una regresión lineal.
        - **Eje X:** `InvestmentEnvironment`, que representa los puntajes del Entorno de Inversión para los países.
        - **Eje Y:** `EconomicQuality`, que representa los puntajes de Calidad Económica para los países.

        **Componentes:**
        - **Puntos de Datos (Scatter Plot):** Representan la relación entre `Entorno de Inversión` y `Calidad Económica` para cada país.
        - **Línea de Regresión (Reg Plot):** Muestra la tendencia general en la relación entre las dos variables. La línea azul indica la mejor línea de ajuste para los datos.

        **Interpretación del Valor de \( R^2 \):**
        El valor de \( R^2 \) indica qué proporción de la variabilidad en la `Calidad Económica` se puede explicar por el `Entorno de Inversión`. Un valor de \( R^2 \) cercano a 1 significa que la línea de regresión explica bien la relación entre `Entorno de Inversión` y `Calidad Económica`, mientras que un valor cercano a 0 indica una relación débil.

        **Ecuación en el Gráfico:**
        En el gráfico, la ecuación de la línea de regresión y el valor de \( R^2 \) están indicados para proporcionar una comprensión visual de la relación entre `Entorno de Inversión` y `Calidad Económica`.
        """)
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='InvestmentEnvironment', y='EconomicQuality', data=df)
        plt.title('Regresión Lineal entre Entorno de Inversión y Calidad Económica')
        plt.xlabel('Entorno de Inversión')
        plt.ylabel('Calidad Económica')
        sns.regplot(x='InvestmentEnvironment', y='EconomicQuality', data=df, scatter=False, color='blue')
        
        # Calcular y mostrar la ecuación
        r2, intercept, coef = calcular_r2(df['InvestmentEnvironment'], df['EconomicQuality'])
        ecuacion = f'$y = {intercept:.2f} + {coef:.2f}x$'
        st.write(f"**Ecuación de la Regresión Lineal:** {ecuacion}")
        st.write(f"**Valor de \( R^2 \):** {r2:.2f}")
        
        # Añadir la ecuación al gráfico
        plt.annotate(ecuacion, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='blue', backgroundcolor='white')
        plt.annotate(f'$R^2 = {r2:.2f}$', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='blue', backgroundcolor='white')
        plt.tight_layout()  # Ajustar diseño
        st.pyplot(plt.gcf())  # Mostrar el gráfico en Streamlit
        plt.clf()  # Limpiar la figura para evitar superposición con otros gráficos

    except Exception as e:
        st.error(f"Error al generar los gráficos: {e}")

def main():
    st.title("Análisis de Datos de Países")

    st.header("Carga de Datos")
    archivo = st.file_uploader("Selecciona un archivo Excel", type=["xls", "xlsx"])

    if archivo:
        df = cargar_datos(archivo)

        if not df.empty:
            df = transformar_datos(df)  # Asegúrate de transformar los datos antes del análisis
            
            total_safety, total_social, total_investment, total_economic = calcular_estadisticas(df)
            
            if total_safety is not None:
                st.header("Estadísticas")
                st.write("País con mayor Seguridad:")
                st.write(total_safety)
                
                st.write("País con mayor Capital Social:")
                st.write(total_social)
                
                st.write("País con mayor Entorno de Inversión:")
                st.write(total_investment)
                
                st.write("País con mayor Calidad Económica:")
                st.write(total_economic)
                
                st.header("Visualización de Datos")
                generar_graficos(df, total_safety, total_social, total_investment, total_economic)
        else:
            st.warning("No se encontraron datos válidos en el archivo.")

if __name__ == "__main__":
    main()
