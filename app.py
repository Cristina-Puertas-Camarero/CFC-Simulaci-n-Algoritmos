import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 Configuración personalizada con identidad visual
st.set_page_config(page_title="Análisis de Datos en Construcción", page_icon="🏗️", layout="wide")

# 📂 Cargar el logo de la empresa
st.image("CFC.png", width=200)  # Ajusta el tamaño si es necesario

# 📂 Definir las páginas en el menú lateral
menu = st.sidebar.radio(
    "Navegación",
    ["🏗️ Presentación y Datos", "🔍 Modelo Predictivo", "🎯 Conclusiones y Perfil"]
)

# 📌 Página 1: Presentación y Datos
if menu == "🏗️ Presentación y Datos":
    st.title("🏗️ Innovación en Construcción a través del Análisis de Datos")
    
    st.subheader("📢 La importancia del Data Science en la industria")
    st.write("""
    La construcción es un sector que enfrenta desafíos constantes en **costos, eficiencia y planificación de obra**. 
    A través del análisis de datos, es posible **mejorar la toma de decisiones**, reducir retrasos y optimizar recursos.  

    Este estudio explora cómo la **ciencia de datos aplicada a la construcción** puede aportar **información clave** 
    para mejorar la ejecución de proyectos, optimizando el rendimiento sin comprometer calidad ni presupuesto.
    """)

    # 🔹 Visualización de datos
    st.subheader("📂 Exploración del Dataset")
    archivo = "datos_sinteticos.xlsx"
    df = pd.read_excel(archivo)
    st.dataframe(df.head())

    # 📊 Distribución de costos
    st.subheader("📊 Análisis de Costos en Proyectos de Construcción")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(df["Costo Total (€)"], bins=10, kde=True, color='blue', ax=ax)
    st.pyplot(fig)
    st.write("""
    **🔎 Interpretación:**  
    - La mayoría de los proyectos tienen costos entre **500,000€ y 2,500,000€**, con algunos alcanzando hasta **4,150,000€**.  
    - Esto muestra que las inversiones en obra pueden **variar enormemente** según el tipo de construcción y materiales empleados.  
    - Comprender esta distribución permite **anticipar costos y optimizar presupuestos** sin comprometer la calidad.  
    """)

    # ⏳ Relación entre duración y costos
    st.subheader("⏳ Impacto de la Duración en Costos")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.scatterplot(x=df["Duración (meses)"], y=df["Costo Total (€)"], hue=df["Tipo de Construcción"], palette="viridis", s=100, ax=ax)
    st.pyplot(fig)
    st.write("""
    **🔎 Interpretación:**  
    - Existe una **relación clara** entre la duración de un proyecto y su costo total.  
    - Los proyectos más largos requieren **mayor inversión**, especialmente en sectores como infraestructura y hospitales.  
    - Esto sugiere que una mejor **planificación y optimización del tiempo** puede traducirse en **reducción de costos**.  
    """)

    # 📈 Eficiencia de materiales
    st.subheader("📈 Relación entre Materiales y Eficiencia en Construcción")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x=df["Material Principal"], y=df["Eficiencia (%)"], palette="Set2", ax=ax)
    st.pyplot(fig)
    st.write("""
    **🔎 Interpretación:**  
    - Diferentes materiales tienen **impactos variados en la eficiencia** del proyecto.  
    - El uso de **hormigón y acero** parece favorecer una mayor eficiencia en obras grandes.  
    - Esto es clave para elegir los **materiales óptimos según el tipo de construcción**, maximizando productividad.  
    """)

    st.write("""
    Estos gráficos demuestran cómo los datos pueden aportar **insights clave** en la planificación de proyectos, optimización de recursos y reducción de costos. 
    A través de modelos predictivos, podemos hacer que estas tendencias sean aún **más precisas y aplicables en la toma de decisiones estratégicas**.
    """)
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 📂 Cargar el modelo guardado
model = joblib.load("modelo_regresion.pkl")  # Asegúrate de que el archivo esté en el mismo directorio

# 📂 Cargar el dataset para obtener opciones únicas
archivo = "datos_sinteticos.xlsx"
df = pd.read_excel(archivo)

# 🔄 Convertir variables categóricas en valores numéricos
label_encoders = {}
categorical_columns = ["Tipo de Construcción", "Material Principal", "Clima Predominante"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convertimos a valores numéricos
    label_encoders[col] = le  # Guardamos los encoders para futuras predicciones

# 📌 Página 2: Modelo Predictivo
st.title("🔍 Predicción de Riesgo de Retraso en Construcción")

st.write("""
En esta sección, exploramos el **modelo de regresión lineal** que estima el **riesgo de retraso (%)** en proyectos de construcción.  
A partir de datos clave como **duración de obra, materiales, tipo de construcción y clima**, el modelo permite realizar simulaciones para optimizar planificación.
""")

# 📥 Entrada de datos para prueba del modelo
st.subheader("📥 Introduce los datos del proyecto")
tipo_construccion = st.selectbox("Tipo de Construcción", label_encoders["Tipo de Construcción"].classes_)
duracion = st.slider("Duración de la obra (meses)", min_value=int(df["Duración (meses)"].min()), 
                     max_value=int(df["Duración (meses)"].max()), value=18)
costo = st.number_input("Costo Total (€)", min_value=int(df["Costo Total (€)"].min()), 
                        max_value=int(df["Costo Total (€)"].max()), value=1500000)
material = st.selectbox("Material Principal", label_encoders["Material Principal"].classes_)
clima = st.selectbox("Clima Predominante", label_encoders["Clima Predominante"].classes_)

# 🔄 Convertir entrada a valores numéricos
tipo_construccion_encoded = label_encoders["Tipo de Construcción"].transform([tipo_construccion])[0]
material_encoded = label_encoders["Material Principal"].transform([material])[0]
clima_encoded = label_encoders["Clima Predominante"].transform([clima])[0]

# 📊 Definir el orden exacto de las columnas según el modelo entrenado
columnas_modelo = ["Proyecto", "Tipo de Construcción", "Duración (meses)", "Costo Total (€)", "Material Principal", 
                   "Clima Predominante", "Eficiencia (%)", "Satisfacción Cliente (1-5)"]

# 📊 Crear un DataFrame con los datos en el mismo orden del modelo entrenado
input_data = pd.DataFrame([[0, tipo_construccion_encoded, duracion, costo, material_encoded, clima_encoded, 
                            80, 4.5]],  # Valores por defecto para las columnas adicionales
                           columns=columnas_modelo)

# 📌 Convertimos los datos al formato correcto
input_data = input_data.astype(float)

# 🔍 Predicción asegurando el orden correcto
try:
    prediccion = model.predict(input_data)[0]
    st.subheader("📈 Predicción del Modelo")
    st.write(f"🔎 **El riesgo estimado de retraso en esta obra es:** {prediccion:.2f}%")

    # 📊 Definir color según el nivel de riesgo
    if prediccion < 10:
        color_texto = "green"  # Riesgo bajo
    elif prediccion < 20:
        color_texto = "orange"  # Riesgo moderado
    else:
        color_texto = "red"  # Riesgo alto

    # 🔎 Mostrar solo el número en grande con el color correspondiente
    st.markdown(
        f"<h1 style='text-align: center; color: {color_texto}; font-size: 80px;'>{prediccion:.2f}%</h1>",
        unsafe_allow_html=True
    )

   
except Exception as e:
    st.error(f"❌ Error al generar la predicción: {e}")

st.write("""
Este análisis permite realizar **simulaciones dinámicas** sobre el impacto de diversas variables en la construcción.  
Con información predictiva, los equipos de obra pueden anticipar y **minimizar imprevistos**, reduciendo costos y optimizando planificación.
""")


# 📌 Página 3: Conclusiones y Perfil Profesional
if menu == "🎯 Conclusiones y Perfil":
    st.title("🎯 Impacto del Data Science en la Industria y Perfil Profesional")

    st.subheader("📊 La importancia del análisis de datos en la optimización empresarial")
    st.write("""
    En el entorno empresarial actual, el **Data Science** es clave para mejorar la precisión en la toma de decisiones.  
    Empresas que implementan **departamentos de datos** pueden aumentar su eficiencia y mejorar su rentabilidad.  

    📈 **Beneficios de un enfoque basado en datos:**  
    - Reducción de costos en planificación y ejecución.  
    - Predicción de tendencias de mercado con mayor precisión.  
    - Optimización de recursos y reducción de riesgos en proyectos.  
    - Aumento de las ventas y mejora en la experiencia del cliente.  
    """)

    # 📊 Gráfico de torta: impacto de un departamento de datos en la empresa
    st.subheader("📊 Impacto de un Departamento de Datos en la Empresa")

    # 🔹 Datos de impacto
    categorias = ["Mejora de precisión en planificación", "Optimización de costos", "Incremento en ventas",
                  "Reducción de riesgos", "Mejora en eficiencia operativa"]
    valores = [25, 20, 30, 15, 10]  # Porcentajes hipotéticos de impacto

    fig, ax = plt.subplots()
    ax.pie(valores, labels=categorias, autopct="%1.1f%%", colors=["#2E86C1", "#1F618D", "#2874A6", "#5499C7", "#85C1E9"])
    ax.set_title("📊 Cómo un Departamento de Datos Impacta la Empresa")

    st.pyplot(fig)

    st.write("""
    Como vemos en el gráfico, un enfoque basado en datos tiene un impacto significativo en la eficiencia y rentabilidad 
    de una empresa. **Construcciones Felipe Castellano** podría optimizar sus procesos con estrategias basadas en datos.  
    """)

    # 💼 Presentación personal
    st.subheader("💼 Sobre mí: Cristina Puertas Camarero")

    st.write("""
    Analista de datos con enfoque en estrategias empresariales, con experiencia en **Data Analysis, Machine Learning** y **Business Intelligence**.  
    Especialista en **interpretación de datos financieros y optimización de procesos** para la toma de decisiones estratégicas.
    """)

    # 📊 Visualización de habilidades en gráfico de barras
    st.subheader("🔍 Habilidades y Nivel de Experiencia")

    habilidades = ["Python", "SQL", "Machine Learning", "Business Intelligence", "Visualización de Datos", 
                   "Análisis", "Estrategia Comercial", "Gestión de Proyectos", "Power BI", "Tableau"]
    niveles = [9, 8, 9, 9, 8, 7, 10, 9, 10, 8]  # Valores de 1 a 10 según experiencia

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(habilidades, niveles, color=["#2E86C1", "#1F618D", "#2874A6", "#5499C7", "#85C1E9", "#154360", "#2980B9","#2874A6", "#5499C7" ])
    ax.set_xlabel("Nivel de Experiencia (1-10)")
    ax.set_title("🔍 Habilidades y Nivel de Experiencia")

    st.pyplot(fig)


