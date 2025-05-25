import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# 📌 Configuración personalizada con identidad visual
st.set_page_config(page_title="Análisis de Datos en Construcción", page_icon="🏗️", layout="wide")

# 📂 Cargar el logo de la empresa
st.image("CFC.png", width=200)  # Ajusta el tamaño si es necesario

# 📂 Definir las páginas en el menú lateral
menu = st.sidebar.radio(
    "Navegación",
    ["Presentación y Datos", "Modelo Predictivo", "Conclusiones y Perfil"]
)
if menu == "Presentación y Datos":
    st.title("Innovación en Construcción a través del Análisis de Datos")

    st.write("""
    La industria de la construcción enfrenta desafíos constantes relacionados con **costos, eficiencia y planificación**.  
    En este contexto, el análisis de datos permite una **toma de decisiones basada en información estratégica**, anticipando riesgos  
    y optimizando recursos para mejorar la rentabilidad y reducir incertidumbre.
    """)

    # 📌 Cargar datos asegurando compatibilidad
    archivo = "datos_sinteticos.xlsx"
    df = pd.read_excel(archivo)  


    # 📊 KPIs clave en construcción
    st.subheader("Indicadores Estratégicos de Construcción")

    col1, col2, col3 = st.columns(3)

    col1.metric(label="Costo Promedio por Proyecto (€)", value=f"{df['Costo Total (€)'].mean():,.0f}")
    col2.metric(label="Duración Promedio (meses)", value=f"{df['Duración (meses)'].mean():,.1f}")
    col3.metric(label="Eficiencia Promedio (%)", value=f"{df['Eficiencia (%)'].mean():,.1f}")

    st.write("""
    **Interpretación:**  
    - **Costo Promedio:** El gasto medio por proyecto es considerable, lo que subraya la importancia de una buena gestión financiera.  
    - **Duración Promedio:** Un análisis detallado del tiempo invertido permite **ajustes en planificación** y reducción de imprevistos.  
    - **Eficiencia Promedio:** La eficiencia global en los proyectos es **clave** para maximizar los recursos y mejorar la rentabilidad.  
    """)

    # 📊 Distribución de costos en construcción
    st.subheader("Distribución de Costos en Construcción")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.histplot(df["Costo Total (€)"], bins=15, kde=True, color='#4CAF50', ax=ax)
    ax.set_xlabel("Costo Total (€)", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title("Análisis de Costos en Construcción", fontsize=14, fontweight="bold")
    st.pyplot(fig)

    st.write("""
    📌 **Hallazgos clave:**  
    - Se observa una gran variabilidad en los costos, indicando diferencias **según el tipo de obra y materiales empleados**.  
    - La presencia de algunos proyectos con costos muy altos sugiere la necesidad de una **gestión de riesgo ajustada**.  
    """)

    # 📌 Relación entre duración y costos
    st.subheader("Impacto de la Duración en Costos Totales")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.regplot(x=df["Duración (meses)"], y=df["Costo Total (€)"], color='#FF9800', scatter_kws={'s':70}, ax=ax)
    ax.set_xlabel("Duración del Proyecto (meses)", fontsize=12)
    ax.set_ylabel("Costo Total (€)", fontsize=12)
    ax.set_title("Relación entre Duración y Costos", fontsize=14, fontweight="bold")
    st.pyplot(fig)

    st.write("""
    📌 **Hallazgos clave:**  
    - Se confirma una **correlación positiva** entre la duración y el costo de los proyectos.  
    - La planificación del tiempo es **crítica** para evitar escaladas innecesarias en los costos finales.  
    """)

    # 📌 Comparación de eficiencia según material principal
    st.subheader("Eficiencia en Construcción según Materiales Utilizados")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x=df["Material Principal"], y=df["Eficiencia (%)"], palette="coolwarm", ax=ax)
    ax.set_xlabel("Material Principal", fontsize=12)
    ax.set_ylabel("Eficiencia (%)", fontsize=12)
    ax.set_title("Impacto del Material en la Eficiencia de Construcción", fontsize=14, fontweight="bold")
    st.pyplot(fig)

    st.write("""
    📌 **Hallazgos clave:**  
    - Se observan variaciones significativas en la eficiencia según el material empleado.  
    - Materiales como **hormigón y acero** ofrecen **mayor estabilidad y rendimiento**, mientras que opciones más económicas  
      pueden comprometer la calidad estructural.  
    """)

    st.write("""
    Este análisis confirma la importancia de una **gestión precisa de costos y materiales**, reforzando la necesidad de modelos predictivos  
    para evaluar riesgos y mejorar la ejecución de los proyectos.  
    """)
# 📌 Evaluación de satisfacción del cliente
    st.subheader("Evaluación de Satisfacción en los Proyectos")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x=df["Tipo de Construcción"], y=df["Satisfacción Cliente (1-5)"], palette="coolwarm", ax=ax)
    ax.set_xlabel("Tipo de Construcción", fontsize=12)
    ax.set_ylabel("Satisfacción Promedio (1-5)", fontsize=12)
    ax.set_title("Nivel de Satisfacción por Tipo de Construcción", fontsize=14, fontweight="bold")
    st.pyplot(fig)

    st.write("""
    📌 **Hallazgos clave:**  
    - Las construcciones **residenciales** tienden a recibir calificaciones más altas en satisfacción.  
    - La satisfacción del cliente es un **indicador clave** para evaluar la calidad del proyecto.  
    """)

    st.write("""
    Este análisis confirma la importancia de una **gestión precisa de costos, materiales y percepción del cliente**, reforzando la necesidad  
    de modelos predictivos para evaluar riesgos y mejorar la ejecución de los proyectos.  
    """)
    ### 📌 **Análisis de rentabilidad por tipo de construcción**
    st.subheader("Rentabilidad Comparativa en Construcción")

    st.write("""
    Este análisis identifica cuáles son las **construcciones más rentables** al comparar el costo total vs. el margen de eficiencia.  
    """)

    # 🔹 Cálculo de rentabilidad
    df["Rentabilidad (%)"] = (df["Eficiencia (%)"] / df["Costo Total (€)"]) * 100
    rentabilidad_por_tipo = df.groupby("Tipo de Construcción")["Rentabilidad (%)"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x="Tipo de Construcción", y="Rentabilidad (%)", data=rentabilidad_por_tipo, palette="viridis", ax=ax)
    ax.set_xlabel("Tipo de Construcción", fontsize=12)
    ax.set_ylabel("Rentabilidad (%)", fontsize=12)
    ax.set_title("Rentabilidad Promedio por Tipo de Construcción", fontsize=14, fontweight="bold")
    st.pyplot(fig)

    st.write("""
    📌 **Hallazgos clave:**  
    - Los proyectos de **infraestructura y comerciales** muestran rentabilidades más bajas debido a costos elevados.  
    - Las construcciones **residenciales** y **industriales** tienen mejores márgenes de rentabilidad.  
    - Un departamento de **Data Science** puede optimizar estrategias de inversión y retorno.  
    """)



    ### 📌 **Impacto del clima y ubicación en la eficiencia**
    st.subheader("Cómo el Clima y la Ubicación Afectan la Construcción")

    st.write("""
    Las condiciones climáticas y la ubicación de un proyecto pueden influir significativamente en la **durabilidad, costos y eficiencia**.  
    """)

    # 🔹 Análisis del impacto del clima en eficiencia
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x=df["Clima Predominante"], y=df["Eficiencia (%)"], palette="coolwarm", ax=ax)
    ax.set_xlabel("Clima", fontsize=12)
    ax.set_ylabel("Eficiencia (%)", fontsize=12)
    ax.set_title("Impacto del Clima en la Eficiencia de Construcción", fontsize=14, fontweight="bold")
    st.pyplot(fig)

    st.write("""
    📌 **Hallazgos clave:**  
    - Las construcciones en **zonas húmedas** tienden a tener menor eficiencia debido a complicaciones estructurales.  
    - Proyectos en **zonas áridas** tienen mayores rendimientos, pero requieren inversiones adicionales en materiales resistentes.  
    - Un equipo de **análisis de datos** puede prever el impacto ambiental y optimizar costos según la ubicación del proyecto.  
    """)



    ### 📌 **Benchmarking y comparación con otras empresas**
    st.subheader("Comparación de Desempeño con Empresas del Sector")

    st.write("""
    El benchmarking permite evaluar el desempeño de los proyectos en comparación con **empresas del mismo sector**,  
    identificando oportunidades de mejora y eficiencia en la construcción.  
    """)

    # 🔹 Generación de datos comparativos ficticios
    competencia = pd.DataFrame({
        "Empresa": ["Constructora A", "Constructora B", "Constructora C", "CFC"],
        "Costo Promedio (€)": [2500000, 2300000, 2800000, df["Costo Total (€)"].mean()],
        "Eficiencia (%)": [78, 82, 76, df["Eficiencia (%)"].mean()],
        "Satisfacción Cliente (1-5)": [4.0, 4.5, 3.8, df["Satisfacción Cliente (1-5)"].mean()]
    })

    st.table(competencia)

    st.write("""
    📌 **Hallazgos clave:**  
    - Nuestra empresa tiene una eficiencia **ligeramente superior** a la competencia, pero costos elevados.  
    - La satisfacción del cliente es **competitiva**, pero aún hay margen para mejorar.  
    - Un departamento de **Data Science** permitiría analizar a fondo cada métrica y mejorar la ventaja competitiva.  
    """)



    ### **Conclusión: ¿Por qué implementar Data Science?**
    st.subheader("¿Cómo un Departamento de Datos puede Transformar la Construcción?")

    st.write("""
    El análisis de datos no es solo un complemento: **es un diferenciador competitivo clave** en la industria de la construcción.  
    Los hallazgos anteriores demuestran que aplicar modelos predictivos y benchmarking en la gestión de proyectos permite:  

    ✔ **Predecir satisfacción del cliente** antes de entregar un proyecto, evitando errores.  
    ✔ **Mejorar rentabilidad** al elegir los tipos de construcción más eficientes.  
    ✔ **Optimizar costos** anticipando el impacto del clima y la ubicación en la obra.  
    ✔ **Comparar el rendimiento con empresas del sector**, detectando oportunidades de mejora.  

    📌 **Invertir en un equipo de Data Science no es un gasto, sino una inversión en competitividad y eficiencia.**  
    Los datos permiten tomar decisiones informadas, reducir riesgos y aumentar la rentabilidad de cada proyecto.
    """)

elif menu == "Modelo Predictivo":
    st.title("Predicción de Riesgo de Retraso en Construcción")

    st.write("""
    La planificación eficiente de proyectos de construcción es clave para evitar sobrecostos y retrasos.  
    Este modelo de **regresión lineal** estima el **riesgo de retraso (%)** a partir de factores como duración de obra, materiales y clima.  
    Con estos análisis, se pueden tomar **decisiones estratégicas** para optimizar tiempos y costos.
    """)

    # 📂 Cargar el modelo de regresión guardado
    try:
        model = joblib.load("modelo_regresion.pkl")  # Asegurar que el archivo está en el mismo directorio
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo del modelo 'modelo_regresion.pkl'.")
        st.stop()

    # 📂 Cargar el dataset para obtener variables únicas
    archivo = "datos_sinteticos.xlsx"
    try:
        df = pd.read_excel(archivo)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{archivo}'.")
        st.stop()

    # 🔄 Convertir variables categóricas en valores numéricos
    label_encoders = {}
    categorical_columns = ["Tipo de Construcción", "Material Principal", "Clima Predominante"]

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Convertimos a valores numéricos
        label_encoders[col] = le  # Guardamos los encoders para futuras predicciones

    # 📥 Entrada de datos para prueba del modelo
    st.subheader("Parámetros del Proyecto")

    tipo_construccion = st.selectbox("Tipo de Construcción", label_encoders["Tipo de Construcción"].classes_)
    duracion = st.slider("Duración de la obra (meses)", int(df["Duración (meses)"].min()), int(df["Duración (meses)"].max()), 18)
    costo = st.number_input("Costo Total (€)", int(df["Costo Total (€)"].min()), int(df["Costo Total (€)"].max()), 1500000)
    material = st.selectbox("Material Principal", label_encoders["Material Principal"].classes_)
    clima = st.selectbox("Clima Predominante", label_encoders["Clima Predominante"].classes_)

    # 🔄 Convertir entrada a valores numéricos
    try:
        tipo_construccion_encoded = label_encoders["Tipo de Construcción"].transform([tipo_construccion])[0]
        material_encoded = label_encoders["Material Principal"].transform([material])[0]
        clima_encoded = label_encoders["Clima Predominante"].transform([clima])[0]
    except KeyError:
        st.error("Error al transformar los datos categóricos. Verifica el dataset y los valores ingresados.")
        st.stop()

    # 📊 Definir el orden exacto de las columnas según el modelo entrenado
    columnas_modelo = ["Proyecto", "Tipo de Construcción", "Duración (meses)", "Costo Total (€)", "Material Principal", 
                       "Clima Predominante", "Eficiencia (%)", "Satisfacción Cliente (1-5)"]

    # 📊 Crear un DataFrame con los datos en el mismo orden del modelo entrenado
    input_data = pd.DataFrame([[0, tipo_construccion_encoded, duracion, costo, material_encoded, clima_encoded, 80, 4.5]], 
                               columns=columnas_modelo)

    # 📌 Convertimos los datos al formato correcto
    input_data = input_data.astype(float)

    # 🔍 Generación de la predicción
    try:
        prediccion = model.predict(input_data)[0]
        st.subheader("Estimación del Riesgo de Retraso")

        # 📊 Definir color según el nivel de riesgo
        if prediccion < 10:
            nivel_riesgo = "**Riesgo Bajo**"
            color = "green"
        elif prediccion < 20:
            nivel_riesgo = "**Riesgo Moderado**"
            color = "orange"
        else:
            nivel_riesgo = "**Riesgo Alto**"
            color = "red"

        # 🔎 Mostrar el resultado visualmente
        st.markdown(f"""
        <div style="text-align: center; font-size: 22px; color: {color};">
            {nivel_riesgo}
            <h1 style="color: {color};">{prediccion:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

        # 📌 Análisis de resultados
        if prediccion < 10:
            st.write("""
            ✔ El proyecto tiene **bajas probabilidades** de retraso según los parámetros ingresados.  
            ✔ Se recomienda seguir con la planificación actual para mantener el rendimiento.  
            """)
        elif prediccion < 20:
            st.write("""
            ⚠️ Existe un **riesgo moderado** de retraso en la obra.  
            ✔ Se sugiere **optimizar tiempos** y revisar el impacto del clima y los materiales en la ejecución.  
            """)
        else:
            st.write("""
            ❌ **Riesgo alto de retraso** identificado.  
            ✔ Se recomienda **revisión completa de la planificación**, ajustes en materiales y tiempos de entrega.  
            ✔ Evaluar estrategias para reducir el impacto en costos y cumplimiento de plazos.  
            """)

    except Exception as e:
        st.error(f"Error al generar la predicción: {e}")

    st.write("""
    La predicción del **riesgo de retraso** permite ajustar planificación, prever costos y minimizar riesgos operativos.  
    Este análisis es una herramienta clave para asegurar la eficiencia y éxito en cada proyecto de construcción.
    """)
elif menu == "Conclusiones y Perfil":
    st.title("Data Science y Business Intelligence: Estrategia para Empresas")

    st.subheader("Cómo los Datos Transforman la Industria")
    st.write("""
    En un mundo impulsado por la información, **Data Science y Business Intelligence** son clave para la toma de decisiones estratégicas.  
    La capacidad de analizar datos permite **reducir costos, mejorar procesos y maximizar rentabilidad**, convirtiendo información en  
    ventajas competitivas reales.  
    """)

    # 📊 **Impacto del Data Science en las Empresas**
    st.subheader("¿Por qué Data Science es clave en la industria?")
    impacto_data = pd.DataFrame({
        "Ámbito": ["Optimización de costos", "Predicción de tendencias", "Reducción de riesgos",
                   "Automatización de procesos", "Mejor toma de decisiones", "Eficiencia operativa"],
        "Impacto (%)": [30, 25, 20, 15, 35, 10]
    })

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x="Impacto (%)", y="Ámbito", data=impacto_data, palette="Blues", ax=ax)
    ax.set_title("Impacto del Data Science en la Empresa", fontsize=14, fontweight="bold")
    ax.set_xlabel("Nivel de Impacto (%)", fontsize=12)
    ax.set_ylabel("")
    st.pyplot(fig)

    st.write("""
    📌 **Conclusiones estratégicas:**  
    ✔ **Optimización de costos** permite mayor margen operativo.  
    ✔ **Automatización de procesos** reduce tiempos de ejecución y errores.  
    ✔ **Predicción de tendencias** mejora la adaptabilidad en mercados volátiles.  
    """)

    # 💼 **Perfil Profesional**
    st.subheader("Cristina Puertas Camarero")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("cris.jpg", width=180)

    with col2:
        st.write("""
        **Data Analyst | Data Scientist | Business Intelligence**  
        Especialista en análisis de datos con orientación estratégica, enfocada en **optimización empresarial, modelos predictivos y rentabilidad**.  
        Mi trayectoria combina **Data Science, Business Intelligence y estrategia comercial**, logrando impacto directo en **decisiones financieras y operativas**.  
        """)

    # 📌 **Enlaces a perfiles profesionales**
    st.markdown("""
    🔗 **LinkedIn:** [Cristina Puertas Camarero](https://www.linkedin.com/in/cristina-puertas-camarero-8955a6349/)  
    🔗 **GitHub:** [Cristina-Puertas-Camarero](https://github.com/Cristina-Puertas-Camarero)  
    📧 **Email:** cris.puertascamarero@gmail.com  
    📍 **Ubicación:** El Puerto de Santa María, Cádiz, España  
    📱 **Teléfono:** [+34] 622 504 007  
    """)

    # 📌 **Habilidades técnicas y blandas visualizadas**
    st.subheader("Habilidades Técnicas y Estratégicas")

    habilidades_df = pd.DataFrame({
        "Habilidad": ["Python (Pandas, NumPy, Matplotlib)", "SQL", "Machine Learning", 
                      "Business Intelligence", "Visualización de Datos (Power BI, Tableau, Streamlit)", 
                      "Análisis Financiero y Rentabilidad", "Modelos Predictivos", "Optimización de Procesos", 
                      "Estrategia Comercial", "Automatización en Microsoft Power Platform", 
                      "Trabajo bajo presión", "Cumplimiento de tiempos de entrega", 
                      "Orientación al cliente y análisis comercial", "Adaptabilidad a cambios", "Resolución de problemas"],
        "Nivel (1-10)": [9, 8, 9, 10, 9, 9, 10, 9, 10, 9, 10, 9, 9, 9, 8]
    })

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x="Nivel (1-10)", y="Habilidad", data=habilidades_df, palette="coolwarm", ax=ax)
    ax.set_title("Mapa de Habilidades en Data Science y Estrategia", fontsize=14, fontweight="bold")
    ax.set_xlabel("Nivel de Dominio (1-10)", fontsize=12)
    ax.set_ylabel("")
    st.pyplot(fig)

    st.write("""
    📌 **Puntos clave:**  
    ✔ **Equilibrio entre habilidades técnicas y estratégicas**, permitiendo tomar decisiones con impacto.  
    ✔ **Trabajo bajo presión y cumplimiento de tiempos**, fundamental en proyectos exigentes.  
    ✔ **Orientación al cliente y comercial**, asegurando alineación con objetivos empresariales.  
    """)

    # 📌 **Experiencia Profesional**
    st.subheader("Experiencia en Datos y Estrategia Empresarial")

    st.write("""
    He trabajado en **entornos empresariales exigentes**, incluyendo el **sector de construcción y planificación de proyectos en Granada**.  
    Durante mi trayectoria, he desarrollado **estrategias basadas en datos para optimizar procesos, reducir costos y mejorar márgenes operativos**.  
    """)

    # 📌 **Proyectos destacados**
    st.subheader("Aplicaciones Reales: Data Science con Propósito")
    proyectos_df = pd.DataFrame({
        "Proyecto": ["A/B Testing - Vanguard", "Trigger Key Words - Suicide Prevention", "Diagnóstico de Cáncer de Mama"],
        "Tecnología Aplicada": ["Pruebas de hipótesis y análisis de datos", "NLP y modelos predictivos", "Machine Learning en clasificación de tumores"],
        "Impacto": ["Mejora en experiencia de usuario", "Prevención temprana de riesgos", "Optimización en diagnóstico médico"]
    })
    st.table(proyectos_df)

    # 📌 **Educación con logos**
    st.subheader("Formación Académica")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("ironhack.jpg", width=180)
        st.write("""
        📌 **Bootcamp en Data Science & Analytics**  
        Ironhack (Feb 2025 - Abr 2025)  
        """)
    
    with col2:
        st.image("uned.jpg", width=180)
        st.write("""
        📌 **Máster en Business Intelligence y Power BI**  
        UNED (Ene 2025 - Jul 2025)  
        """)

    # 📌 **Cierre persuasivo**
    st.subheader("📢 ¿Por qué Data Science es clave?")
    st.write("""
    En la actualidad, **Data Science define las estrategias empresariales exitosas**.  
    Mi combinación de **análisis de datos, visión estratégica y optimización comercial** me permite contribuir al crecimiento de cualquier empresa.  

    ✔ **Transformación de procesos mediante datos reales y accionables.**  
    ✔ **Optimización de márgenes financieros y reducción de costos.**  
    ✔ **Modelos predictivos que anticipan tendencias y riesgos empresariales.**  

    📌 ¿Por qué Construcciones Felipe Castellano?

    Construcciones Felipe Castellano representa el compromiso con la innovación en construcción y la integración de soluciones estratégicas basadas en datos. Su enfoque en optimización de proyectos, eficiencia operativa y calidad estructural es exactamente el tipo de entorno donde mi perfil en Data Science y Business Intelligence puede generar un impacto tangible.
    Lo que más me motiva de la empresa es su visión orientada a la mejora continua, el uso de datos para anticipar desafíos en planificación y su apuesta por maximizar la rentabilidad sin comprometer calidad. Mi experiencia en análisis financiero, modelos predictivos y optimización de procesos puede contribuir a fortalecer la gestión de costos y tiempos de entrega, asegurando una ejecución más eficiente de los proyectos
    Trabajar en un sector exigente como la construcción y aplicar técnicas de Data Science es un reto apasionante, donde cada decisión basada en datos puede significar una mejor planificación, menor riesgo y mayor competitividad. Estoy convencida de que puedo aportar valor real en este contexto y ayudar a transformar la industria con información estratégica.

    📌 **. Estoy lista para nuevos desafíos. Muchas gracias por esta oportunidad de presentación de mi perfil**  
    """)
