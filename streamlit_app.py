import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Dengue App", layout="wide")

st.title("🦟 Dengue App")

# Crear tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Inicio y explicación", "📖 Uso de la app", "📊 Métricas", "📈 Dashboards"])

with tab1:
    st.header("🦟 Inicio y Explicación")
    
    st.markdown("---")
    
    st.subheader("📋 Descripción General")
    st.write("""
    Esta aplicación forma parte de un **Proyecto Integrador de Ciencia de Datos** y tiene como objetivo 
    construir un sistema de análisis de dengue que permite explorar, clasificar y analizar casos de dengue 
    en Argentina según sus características epidemiológicas, climáticas y demográficas.
    """)
    
    st.subheader("📊 Datos Utilizados")
    st.write("""
    El análisis se basa en un dataset enriquecido que contiene:
    - **Casos de dengue**: Datos de casos confirmados en diferentes provincias y departamentos
    - **Información temporal**: Años, semanas epidemiológicas y fechas de registro
    - **Variables climáticas**: Temperatura, humedad, precipitación
    - **Datos demográficos**: Población, densidad poblacional, superficie
    - **Clasificación etaria**: Distribución de casos por grupo de edad
    """)
    
    st.subheader("🔬 Proceso de Desarrollo")
    st.write("""
    El análisis se divide en distintas etapas:
    
    1. **Extracción y procesamiento de datos**: Obtención de datos desde múltiples fuentes
    2. **Análisis exploratorio**: EDA con hipótesis sobre la relación clima-dengue
    3. **Análisis estadístico**: 
       - Relación entre temperatura, humedad, precipitación y casos
       - Variación regional por zona climática (Subtropical, Templada, Árida, Fría)
       - Impacto de densidad poblacional
    4. **Análisis temporal**: Identificación de picos y patrones estacionales
    5. **Análisis etario**: Influencia de grupos de edad en la distribución de casos
    """)
    
    st.subheader("📈 Hallazgos Principales")
    st.write("""
    - **Estacionalidad**: Mayor concentración de casos en meses cálidos (febrero, marzo, abril)
    - **Variabilidad regional**: La relación clima-dengue varía significativamente por región
    - **Zona Subtropical**: Muestra correlación casi lineal entre densidad poblacional y casos
    - **Zona Templada**: Comportamiento proporcional pero con saturación en densidades altas
    - **Grupos etarios**: Mayor incidencia en población de 5 a 24 años (aproximadamente 8% por zona)
    """)
    
    st.subheader("🎯 Objetivo de la App")
    st.write("""
    Esta plataforma interactiva permite:
    - Explorar el espacio de datos de dengue de forma visual
    - Analizar métricas agregadas y tendencias
    - Visualizar dashboards con insights clave
    - Comprender los patrones epidemiológicos del dengue en Argentina
    """)

with tab2:
    st.header("📖 Uso de la App")
    
    st.markdown("---")
    
    st.subheader("🔮 Predictor de Casos de Dengue")
    st.write("Utiliza el modelo entrenado para predecir la cantidad de casos de dengue basándote en parámetros climáticos y epidemiológicos.")
    
    st.markdown("---")
    
    # Crear dos columnas para los inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Datos Temporales")
        
        # Selector de mes y día
        mes = st.selectbox(
            "Mes",
            options=list(range(1, 13)),
            format_func=lambda x: ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                                   "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"][x-1],
            key="mes"
        )
        
        dia = st.slider(
            "Día",
            min_value=1,
            max_value=28,
            value=1,
            key="dia"
        )
        
        # Calcular automáticamente la semana epidemiológica
        try:
            fecha = datetime(2025, mes, dia)
            semana_epidemiologica = fecha.isocalendar()[1]
        except ValueError:
            semana_epidemiologica = 1
        
        st.info(f"📍 Semana epidemiológica calculada: **{semana_epidemiologica}**")
    
    with col2:
        st.subheader("🌡️ Variables Climáticas")
        
        temperatura = st.number_input(
            "Temperatura semanal promedio (°C)",
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            step=0.1,
            key="temp"
        )
        
        humedad = st.number_input(
            "Humedad semanal promedio (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1,
            key="hum"
        )
        
        precipitacion = st.number_input(
            "Precipitación semanal promedio (mm)",
            min_value=0.0,
            max_value=500.0,
            value=10.0,
            step=0.1,
            key="prec"
        )
    
    st.markdown("---")
    
    st.subheader("👥 Datos Demográficos y Adicionales")
    
    col3, col4 = st.columns(2)
    
    with col3:
        densidad = st.number_input(
            "Densidad del departamento (personas/km²)",
            min_value=0.0,
            max_value=50000.0,
            value=5000.0,
            step=10.0,
            key="dens"
        )
    
    with col4:
        st.info("ℹ️ El año se fija automáticamente a 2025.")
    
    # Año fijo a 2025
    ano = 2025
    
    st.markdown("---")
    
    st.subheader("📊 Casos Previos (Lags)")
    st.write("Ingresa los casos de dengue de semanas anteriores para mejorar la predicción:")
    
    col_lag1, col_lag2, col_lag4 = st.columns(3)
    
    with col_lag1:
        lag1 = st.number_input(
            "Casos hace 1 semana (lag1)",
            min_value=0,
            max_value=10000,
            value=10,
            key="lag1"
        )
    
    with col_lag2:
        lag2 = st.number_input(
            "Casos hace 2 semanas (lag2)",
            min_value=0,
            max_value=10000,
            value=8,
            key="lag2"
        )
    
    with col_lag4:
        lag4 = st.number_input(
            "Casos hace 4 semanas (lag4)",
            min_value=0,
            max_value=10000,
            value=5,
            key="lag4"
        )
    
    st.markdown("---")
    
    # Botón para hacer la predicción
    st.subheader("📊 Datos a Predecir")
    
    # Mostrar tabla con los parámetros
    datos_prediccion = pd.DataFrame({
        "Parámetro": ["Año", "Densidad", "Temperatura (°C)", "Humedad (%)", "Precipitación (mm)", "Lag 1 (semana -1)", "Lag 2 (semana -2)", "Lag 4 (semana -4)"],
        "Valor": [str(ano), f"{densidad:.2f}", f"{temperatura:.2f}", f"{humedad:.2f}", f"{precipitacion:.2f}", str(lag1), str(lag2), str(lag4)]
    })
    
    st.dataframe(datos_prediccion, use_container_width=True)
    
    # Fecha seleccionada
    fecha_seleccionada = f"2025-{str(mes).zfill(2)}-{str(dia).zfill(2)}"
    st.caption(f"Fecha seleccionada: {fecha_seleccionada} — semana epidemiológica: {semana_epidemiologica}")
    
    # Botón de predicción
    if st.button("🔮 Predecir", use_container_width=True, type="primary"):
        try:
            # Cargar el modelo
            with open("/workspaces/Dengue-app/Modelo/model.pkl", "rb") as f:
                modelo = pickle.load(f)
            
            # Preparar datos para la predicción en el orden correcto
            # Orden esperado: ['anio', 'temp_sem_prom', 'hum_sem_prom', 'prec_sem_prom', 'densidad', 'lag1', 'lag2', 'lag4']
            X_pred = np.array([[ano, temperatura, humedad, precipitacion, densidad, lag1, lag2, lag4]])
            
            # Hacer la predicción
            prediccion_proba = modelo.predict_proba(X_pred)[0]
            prediccion_clase = modelo.predict(X_pred)[0]
            confianza = max(prediccion_proba) * 100
            probabilidad_casos = prediccion_proba[prediccion_clase]
            
            # Definir rangos según la clase
            if prediccion_clase == 0:
                rango_texto = "0 a 49 casos"
                casos_minimo, casos_maximo = 0, 49
                riesgo_nivel = "BAJO"
            elif prediccion_clase == 1:
                rango_texto = "49 a 199 casos"
                casos_minimo, casos_maximo = 49, 199
                riesgo_nivel = "MEDIO"
            else:
                rango_texto = "199 a 200+ casos"
                casos_minimo, casos_maximo = 199, 200
                riesgo_nivel = "ALTO"
            
            # Mostrar resultado
            st.markdown("---")
            st.subheader("✅ Resultado de la Predicción")
            
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.metric("Rango de Casos", rango_texto)
            with col_result2:
                st.metric("Probabilidad", f"{probabilidad_casos * 100:.1f}%")
            with col_result3:
                st.metric("Confianza", f"{confianza:.1f}%")
            
            st.info(f"📈 Para la semana epidemiológica {semana_epidemiologica}, el modelo predice un rango de **{rango_texto}** (Riesgo: **{riesgo_nivel}**) con confianza de {confianza:.1f}%.")
            
        except FileNotFoundError:
            st.error("❌ El archivo del modelo no se encontró. Por favor, asegúrate de que model.pkl exista en la carpeta Modelo/.")
        except Exception as e:
            st.error(f"❌ Error al realizar la predicción: {str(e)}")

with tab3:
    st.header("📊 Métricas")
    st.markdown("---")
    
    # Cargar datos para métricas (mismo preprocesamiento básico que en dashboards)
    import os
    file_path = os.path.join("info", "dengue_enriched_final.xlsx")
    df_metrics = pd.read_excel(file_path)
    
    # Asegurar columna de casos
    case_col = next((c for c in ["cantidad_casos","casos","n_casos","count_casos"] if c in df_metrics.columns), None)
    if case_col is None:
        case_col = "cantidad_casos"
    
    # Normalizar provincia
    def fix_prov_name_simple(p):
        if pd.isna(p): return p
        p = str(p).strip().upper()
        if p in {"CABA","CIUDAD AUTONOMA BUENOS AIRES","CAPITAL FEDERAL","CIUDAD AUTONOMA DE BUENOS AIRES"}:
            return "CIUDAD AUTONOMA DE BUENOS AIRES"
        repl = str.maketrans("ÁÉÍÓÚÑ", "AEIOUN")
        return p.translate(repl)
    
    if "provincia_nombre" in df_metrics.columns:
        df_metrics["provincia_nombre"] = df_metrics["provincia_nombre"].apply(fix_prov_name_simple)
    
    # Mapeo provincia -> clima_region
    PROVINCIA_A_CLIMA = {
        "BUENOS AIRES":"TEMPLADO","CIUDAD AUTONOMA DE BUENOS AIRES":"TEMPLADO","CABA":"TEMPLADO",
        "ENTRE RIOS":"TEMPLADO","SANTA FE":"TEMPLADO","CORDOBA":"TEMPLADO","LA PAMPA":"TEMPLADO",
        "MISIONES":"SUBTROPICAL","CHACO":"SUBTROPICAL","CORRIENTES":"SUBTROPICAL","FORMOSA":"SUBTROPICAL","TUCUMAN":"SUBTROPICAL",
        "CATAMARCA":"ARIDO/SEMIARIDO","LA RIOJA":"ARIDO/SEMIARIDO","SAN JUAN":"ARIDO/SEMIARIDO",
        "SAN LUIS":"ARIDO/SEMIARIDO","SANTIAGO DEL ESTERO":"ARIDO/SEMIARIDO","SANTA CRUZ":"ARIDO/SEMIARIDO",
        "TIERRA DEL FUEGO, ANTARTIDA E ISLAS DEL ATLANTICO SUR":"ARIDO/SEMIARIDO","TIERRA DEL FUEGO":"ARIDO/SEMIARIDO",
        "MENDOZA":"FRIO/MONTANA","NEUQUEN":"FRIO/MONTANA","RIO NEGRO":"FRIO/MONTANA",
        "CHUBUT":"FRIO/MONTANA","JUJUY":"FRIO/MONTANA","SALTA":"FRIO/MONTANA",
    }
    
    if "provincia_nombre" in df_metrics.columns:
        df_metrics["clima_region"] = df_metrics["provincia_nombre"].map(PROVINCIA_A_CLIMA).fillna("TEMPLADO")
    else:
        df_metrics["clima_region"] = "TEMPLADO"
    
    # Asegurar casos numéricos
    if case_col in df_metrics.columns:
        df_metrics[case_col] = pd.to_numeric(df_metrics[case_col], errors="coerce").fillna(0).clip(lower=0)
    
    # --- KPI 1: Total de casos ---
    total_casos = df_metrics[case_col].sum()
    
    # --- KPI 2: Casos promedio por semana (agregado a nivel semanal) ---
    if "anio" in df_metrics.columns and "semana_epidemiologica" in df_metrics.columns:
        casos_por_semana = df_metrics.groupby(["anio","semana_epidemiologica"], as_index=False)[case_col].sum()
        prom_casos_semana = casos_por_semana[case_col].mean()
    else:
        prom_casos_semana = df_metrics[case_col].mean()
    
    # --- KPI 3: Región más afectada ---
    casos_por_region = df_metrics.groupby("clima_region")[case_col].sum().sort_values(ascending=False)
    region_mas_afectada = casos_por_region.index[0] if len(casos_por_region) > 0 else "N/A"
    casos_region_max = casos_por_region.iloc[0] if len(casos_por_region) > 0 else 0
    
    # --- KPI 4: Temperatura y humedad promedio ---
    temp_prom = df_metrics[[c for c in df_metrics.columns if c.lower().startswith("temp")]].values.mean()
    hum_prom = df_metrics[[c for c in df_metrics.columns if c.lower().startswith("hum")]].values.mean()
    
    # --- KPI 5: Provincia más afectada ---
    if "provincia_nombre" in df_metrics.columns:
        casos_por_prov = df_metrics.groupby("provincia_nombre")[case_col].sum().sort_values(ascending=False)
        prov_mas_afectada = casos_por_prov.index[0] if len(casos_por_prov) > 0 else "N/A"
        casos_prov_max = casos_por_prov.iloc[0] if len(casos_por_prov) > 0 else 0
    else:
        prov_mas_afectada = "N/A"
        casos_prov_max = 0
    
    # --- KPI 6: Distribución por grupo etario ---
    if "grupo_edad_desc" in df_metrics.columns:
        casos_por_edad = df_metrics.groupby("grupo_edad_desc")[case_col].sum().sort_values(ascending=False)
        edad_mas_afectada = casos_por_edad.index[0] if len(casos_por_edad) > 0 else "N/A"
        casos_edad_max = casos_por_edad.iloc[0] if len(casos_por_edad) > 0 else 0
    else:
        edad_mas_afectada = "N/A"
        casos_edad_max = 0
    
    # --- Mostrar KPIs en columnas ---
    st.subheader("🎯 Indicadores Clave (KPIs)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Casos", f"{int(total_casos):,}")
    with col2:
        st.metric("Casos Promedio por Semana", f"{prom_casos_semana:.1f}")
    with col3:
        st.metric("Temperatura Promedio", f"{temp_prom:.1f}°C")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric("Humedad Promedio", f"{hum_prom:.1f}%")
    with col5:
        st.metric("Región Más Afectada", region_mas_afectada, delta=f"{int(casos_region_max):,} casos")
    with col6:
        st.metric("Provincia Más Afectada", prov_mas_afectada, delta=f"{int(casos_prov_max):,} casos")
    
    st.markdown("---")
    
    # --- Gráficos de contexto ---
    st.subheader("📈 Análisis Complementarios")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.write("**Casos por Región**")
        if not casos_por_region.empty:
            df_region = casos_por_region.reset_index()
            df_region.columns = ["clima_region", "casos"]
            chart_region = alt.Chart(df_region).mark_bar().encode(
                x=alt.X("clima_region:N", title="Región"),
                y=alt.Y("casos:Q", title="Total de Casos"),
                color=alt.Color("clima_region:N", legend=None)
            ).properties(width=300, height=250)
            st.altair_chart(chart_region, use_container_width=True)
    
    with col_g2:
        st.write("**Casos por Provincia (Top 10)**")
        if not casos_por_prov.empty:
            df_prov = casos_por_prov.head(10).reset_index()
            df_prov.columns = ["provincia_nombre", "casos"]
            chart_prov = alt.Chart(df_prov).mark_barh().encode(
                y=alt.Y("provincia_nombre:N", title="Provincia", sort="-x"),
                x=alt.X("casos:Q", title="Total de Casos"),
                color=alt.Color("provincia_nombre:N", legend=None)
            ).properties(width=300, height=300)
            st.altair_chart(chart_prov, use_container_width=True)
    
    st.markdown("---")
    
    # --- Serie temporal por región ---
    if "anio" in df_metrics.columns and "semana_epidemiologica" in df_metrics.columns:
        st.subheader("📅 Evolución Temporal por Región")
        ts_data = df_metrics.groupby(["anio","semana_epidemiologica","clima_region"], as_index=False)[case_col].sum()
        ts_data["fecha_id"] = ts_data["anio"].astype(str) + "-W" + ts_data["semana_epidemiologica"].astype(str)
        
        ts_chart = alt.Chart(ts_data).mark_line(point=True).encode(
            x=alt.X("fecha_id:N", title="Año-Semana"),
            y=alt.Y(f"{case_col}:Q", title="Casos"),
            color=alt.Color("clima_region:N", title="Región"),
            tooltip=["fecha_id:N","clima_region:N",f"{case_col}:Q"]
        ).properties(width=800, height=300, title="Casos por Región a lo Largo del Tiempo")
        
        st.altair_chart(ts_chart, use_container_width=True)
    
    st.markdown("---")
    st.caption("*Métricas calculadas en tiempo real desde `info/dengue_enriched_final.xlsx`*")

with tab4:
    st.header("📈 Dashboards")
    st.markdown("---")
    st.subheader("Huella por región (z-score por variable)")
    import altair as alt
    import pandas as pd
    import numpy as np
    import re
    from datetime import date
    import os

    # --- Cargar y preprocesar el archivo real ---
    file_path = os.path.join("info", "dengue_enriched_final.xlsx")
    df = pd.read_excel(file_path)

    # Normalización de texto (mantener utilidades ya definidas arriba)
    def normalize_text(s):
        if pd.isna(s): return s
        s = str(s).strip()
        s = (s.replace("Ã‘", "Ñ")
             .replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
             .replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U"))
        s = re.sub(r'\s+', ' ', s)
        return s.upper()

    def fix_prov_name_simple(p):
        if pd.isna(p): return p
        p = str(p).strip().upper()
        if p in {"CABA","CIUDAD AUTONOMA BUENOS AIRES","CAPITAL FEDERAL","CIUDAD AUTONOMA DE BUENOS AIRES"}:
            return "CIUDAD AUTONOMA DE BUENOS AIRES"
        # Normalizar tildes básicas
        repl = str.maketrans("ÁÉÍÓÚÑ", "AEIOUN")
        return p.translate(repl)

    if "provincia_nombre" in df.columns:
        df["provincia_nombre"] = df["provincia_nombre"].apply(fix_prov_name_simple)

    if "departamento_nombre" in df.columns:
        df["departamento_nombre"] = df["departamento_nombre"].apply(lambda x: normalize_text(x) if pd.notna(x) else x)

    # Forzar numéricos en columnas climáticas y demográficas
    maybe_numeric = [c for c in df.columns if any(k in c.lower() for k in ["lat","lon","temp","hum","prec","poblacion","densidad","superficie"])]
    for c in maybe_numeric:
        if c in df.columns and df[c].dtype == "O":
            df[c] = pd.to_numeric(df[c].apply(lambda x: str(x).strip() if pd.notna(x) else x).replace('', pd.NA), errors="coerce")

    # Asegurar columna de casos
    case_col = next((c for c in ["cantidad_casos","casos","n_casos","count_casos"] if c in df.columns), None)
    if case_col is not None:
        df[case_col] = pd.to_numeric(df[case_col], errors="coerce").fillna(0).clip(lower=0)

    # Fecha semanal (si aún no existe)
    def iso_week_start_safe(year, week):
        try: return pd.to_datetime(date.fromisocalendar(int(year), int(week), 1))
        except: return pd.NaT
    if "fecha_semana" not in df.columns:
        if {"anio","semana_epidemiologica"}.issubset(df.columns):
            df["fecha_semana"] = df.apply(lambda r: iso_week_start_safe(r.get("anio"), r.get("semana_epidemiologica")), axis=1)
        elif "fecha" in df.columns:
            df["fecha_semana"] = pd.to_datetime(df["fecha"], errors="coerce")
        else:
            df["fecha_semana"] = pd.NaT

    # -------- Promedio semanal intrafila de clima (temp/hum/prec) --------
    dias = ["_L","_M","_X","_J","_V","_S","_D"]
    for base in ["temp","hum","prec"]:
        cols = [c for c in df.columns if c.lower().startswith(base + "_") and any(c.endswith(d) for d in dias)]
        if cols:
            df[f"{base}_sem_prom"] = df[cols].mean(axis=1)
        else:
            # fallback: usar cualquier columna que comience con la raíz (p.ej. temp_L sin guion)
            cand = [c for c in df.columns if c.lower().startswith(base)]
            if cand:
                df[f"{base}_sem_prom"] = df[cand].mean(axis=1)

    # -------- Mapeo provincia -> clima_region (copiado del TP2) --------
    PROVINCIA_A_CLIMA = {
        # TEMPLADO
        "BUENOS AIRES": "TEMPLADO",
        "CIUDAD AUTONOMA DE BUENOS AIRES": "TEMPLADO",
        "CABA": "TEMPLADO",
        "ENTRE RIOS": "TEMPLADO",
        "SANTA FE": "TEMPLADO",
        "CORDOBA": "TEMPLADO",
        "LA PAMPA": "TEMPLADO",

        # SUBTROPICAL (NEA)
        "MISIONES": "SUBTROPICAL",
        "CHACO": "SUBTROPICAL",
        "CORRIENTES": "SUBTROPICAL",
        "FORMOSA": "SUBTROPICAL",
        "TUCUMAN": "SUBTROPICAL",

        # ARIDO/SEMIARIDO
        "CATAMARCA": "ARIDO/SEMIARIDO",
        "LA RIOJA": "ARIDO/SEMIARIDO",
        "SAN JUAN": "ARIDO/SEMIARIDO",
        "SAN LUIS": "ARIDO/SEMIARIDO",
        "SANTIAGO DEL ESTERO": "ARIDO/SEMIARIDO",
        "SANTA CRUZ": "ARIDO/SEMIARIDO",
        "TIERRA DEL FUEGO, ANTARTIDA E ISLAS DEL ATLANTICO SUR": "ARIDO/SEMIARIDO",
        "TIERRA DEL FUEGO": "ARIDO/SEMIARIDO",

        # FRIO/MONTANA
        "MENDOZA": "FRIO/MONTANA",
        "NEUQUEN": "FRIO/MONTANA",
        "RIO NEGRO": "FRIO/MONTANA",
        "CHUBUT": "FRIO/MONTANA",
        "JUJUY": "FRIO/MONTANA",
        "SALTA": "FRIO/MONTANA",
    }

    if "provincia_nombre" in df.columns:
        df["clima_region"] = df["provincia_nombre"].map(PROVINCIA_A_CLIMA)
        provincias_en_df = set(df["provincia_nombre"].dropna().unique())
        provincias_mapeadas = set(PROVINCIA_A_CLIMA.keys())
        faltantes = sorted(p for p in provincias_en_df if p not in provincias_mapeadas)
        if faltantes:
            # fallback operativo
            df.loc[df["provincia_nombre"].isin(faltantes), "clima_region"] = "TEMPLADO"
        df["clima_region"] = df["clima_region"].fillna("TEMPLADO")
    else:
        df["clima_region"] = "TEMPLADO"

    # Crear columnas auxiliares si faltan
    if "densidad" not in df.columns:
        if "poblacion" in df.columns and "superficie" in df.columns:
            df["densidad"] = pd.to_numeric(df["poblacion"], errors="coerce") / pd.to_numeric(df["superficie"], errors="coerce")
        else:
            df["densidad"] = pd.NA

    # Agrupación para dashboards: asegurar que existan las columnas canónicas
    reg_col = "clima_region"
    features = ["densidad", "temp_sem_prom", "hum_sem_prom", "prec_sem_prom"]

    use_cols = [c for c in ["provincia_nombre","departamento_nombre","fecha_semana", reg_col] if c in df.columns] + [f for f in features if f in df.columns]
    if len(use_cols) < 4:
        st.warning("No se detectaron todas las columnas necesarias para los dashboards; se mostrarán los gráficos disponibles con las columnas existentes.")

    dfv = df[use_cols].copy()
    # Drop rows donde región o features clave falten
    key_required = [reg_col] + [f for f in features if f in dfv.columns]
    dfv = dfv.dropna(subset=[k for k in key_required if k in dfv.columns])
    dfv = dfv.rename(columns={reg_col: "region"})
    dfv["fecha_semana"] = pd.to_datetime(dfv["fecha_semana"], errors="coerce")
    meses_map = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    dfv["mes_desc"] = dfv["fecha_semana"].dt.month.map(meses_map)
    # --- Graficar ---
    regiones = sorted(dfv["region"].dropna().unique().tolist())
    # Determinar features disponibles (evita errores si faltan columnas)
    available_features = [f for f in features if f in dfv.columns]
    if not available_features:
        st.warning("No se detectaron variables climáticas/demográficas disponibles para graficar.")
    else:
        # Heatmap
        mean_by_reg_feat = (dfv.melt(id_vars=["region"], value_vars=available_features, var_name="variable", value_name="valor")
            .groupby(["region","variable"], as_index=False)["valor"].mean())
        z_df = []
        for v in available_features:
            sub = mean_by_reg_feat[mean_by_reg_feat["variable"] == v].copy()
            mu, sd = sub["valor"].mean(), sub["valor"].std(ddof=0)
            sub["z"] = (sub["valor"] - mu) / (sd if sd > 0 else 1.0)
            z_df.append(sub)
        z_df = pd.concat(z_df, ignore_index=True)
        order_vars = (z_df.groupby("variable")["z"].apply(lambda s: s.max()-s.min())
            .sort_values(ascending=False).index.tolist())
        heat = (alt.Chart(z_df)
            .mark_rect()
            .encode(
                x=alt.X("variable:N", title="Variable", sort=order_vars),
                y=alt.Y("region:N", title="Región"),
                color=alt.Color("z:Q", title="Z-score", scale=alt.Scale(scheme="blueorange", domainMid=0)),
                tooltip=["region:N","variable:N",alt.Tooltip("valor:Q",format=".2f"),alt.Tooltip("z:Q",format=".2f")]
            )
            .properties(title="Huella por región (z-score por variable)", width=420, height=160)
        )
        st.altair_chart(heat, use_container_width=True)
    st.markdown("---")
    st.subheader("Distribución de variables por región")
    # Preparar id_vars existentes para boxplots
    id_vars = ["region"] + [c for c in ["provincia_nombre","departamento_nombre","mes_desc"] if c in dfv.columns]
    long_df = pd.DataFrame()
    if available_features:
        long_df = dfv.melt(id_vars=id_vars, value_vars=available_features, var_name="variable", value_name="valor").dropna(subset=["valor"])
    else:
        long_df = pd.DataFrame()
    box = (alt.Chart(long_df)
        .mark_boxplot(outliers=True)
        .encode(
            y=alt.Y("region:N", title="Región", sort=regiones),
            x=alt.X("valor:Q", title="Valor"),
            color=alt.Color("region:N", legend=None),
            tooltip=["region:N","variable:N","valor:Q"]
        )
        .properties(width=250, height=120)
    )
    if not long_df.empty:
        box_grid = (box.facet(column=alt.Column("variable:N", title=None, sort=order_vars))
            .resolve_scale(x="independent")
            .properties(title="Distribución de variables por región")
        )
        st.altair_chart(box_grid, use_container_width=True)
    else:
        st.info("No hay datos suficientes para mostrar el boxplot de variables por región.")
    st.caption("*Datos reales procesados del archivo dengue_enriched_final.xlsx*")

    # Opción de depuración: mostrar vista previa de dfv y columnas derivadas
    if st.checkbox("Mostrar vista previa de datos (debug)"):
        st.write("Columnas disponibles en `dfv`:", dfv.columns.tolist())
        st.dataframe(dfv.head(10))
        if available_features:
            st.write("Descriptivo de features disponibles:")
            st.dataframe(dfv[available_features].describe(include='all'))
