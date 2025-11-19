import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
import re
import os
from datetime import date

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
    
    """)
    
    st.subheader("📈 Hallazgos Principales")
    st.write("""
    - **Estacionalidad**: Mayor concentración de casos en meses cálidos (febrero, marzo, abril)
    - **Variabilidad regional**: La relación clima-dengue varía significativamente por región
    - **Zona Subtropical**: Muestra correlación casi lineal entre densidad poblacional y casos
    - **Zona Templada**: Comportamiento proporcional pero con saturación en densidades altas
    
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
        
        dia = st.number_input(
            "Día",
            min_value=1,
            max_value=28,
            value=1,
            step=1,
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
        
        temperatura = st.slider(
            "Temperatura semanal promedio (°C)",
            min_value=-15.0,
            max_value=57.0,
            value=20.0,
            key="temp"
        )
        
        humedad = st.slider(
            "Humedad semanal promedio (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            key="hum"
        )
        
        precipitacion = st.slider(
            "Precipitación semanal promedio (mm)",
            min_value=0.0,
            max_value=400.0,
            value=10.0,
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
    
    st.subheader("📊 Casos Previos ")
    st.write("Ingresa los casos de dengue de semanas anteriores para mejorar la predicción:")
    
    col_lag1, col_lag2, col_lag4 = st.columns(3)
    
    with col_lag1:
        lag1 = st.number_input(
            "Casos hace 1 semana ",
            min_value=0,
            max_value=10000,
            value=10,
            key="lag1"
        )
    
    with col_lag2:
        lag2 = st.number_input(
            "Casos hace 2 semanas ",
            min_value=0,
            max_value=10000,
            value=8,
            key="lag2"
        )
    
    with col_lag4:
        lag4 = st.number_input(
            "Casos hace 4 semanas ",
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
            # Cargar el modelo (ruta relativa para compatibilidad con Streamlit Cloud)
            model_path = os.path.join("Modelo", "model.pkl")
            with open(model_path, "rb") as f:
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
                rango_texto = "200 o mas casos"
                casos_minimo, casos_maximo = 200, 9999
                riesgo_nivel = "ALTO"
            
            # Mostrar resultado
            st.markdown("---")
            st.subheader("✅ Resultado de la Predicción")
            
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.metric("Rango de Casos", rango_texto)
            with col_result2:
                st.metric("Probabilidad", f"{probabilidad_casos * 100:.1f}%")
         #   with col_result3:
           #     st.metric("Confianza", f"{confianza:.1f}%")
            
            st.info(f"📈 Para la semana epidemiológica {semana_epidemiologica}, el modelo predice un rango de **{rango_texto}** (Riesgo: **{riesgo_nivel}**) con confianza de {confianza:.1f}%.")
            
        except FileNotFoundError:
            st.error("❌ El archivo del modelo no se encontró. Por favor, asegúrate de que model.pkl exista en la carpeta Modelo/.")
        except Exception as e:
            st.error(f"❌ Error al realizar la predicción: {str(e)}")

with tab3:
    st.header("📊 Métricas")
    st.markdown("---")
    st.markdown("""
    <span style='font-size:1.1em'><b>¿Por qué estas métricas?</b></span>
    
    Las métricas seleccionadas permiten analizar el comportamiento del dengue en Argentina desde diferentes perspectivas:
    
    - <b>Casos totales y promedio semanal:</b> muestran la magnitud y tendencia general de la epidemia.
    - <b>Temperatura, humedad y precipitación:</b> son variables climáticas clave que influyen en la proliferación del mosquito transmisor y la dinámica de los brotes.
    - <b>Densidad poblacional:</b> refleja el potencial de transmisión en áreas urbanas y rurales.
    - <b>Lags epidemiológicos:</b> los casos de semanas previas ayudan a entender el efecto de la inercia y el arrastre en la evolución de los contagios.
    
    Además, se analizan correlaciones globales para visualizar el impacto relativo de cada variable sobre los casos, facilitando la interpretación y la toma de decisiones.
    """, unsafe_allow_html=True)
    
    # ===== PREPROCESAMIENTO PARA MÉTRICAS (basado en TP2/TP3) =====
    file_path = os.path.join("info", "dengue_enriched_final.xlsx")
    df_m = pd.read_excel(file_path)
    
    # Columna de casos
    case_col = next((c for c in ["cantidad_casos","casos","n_casos","count_casos"] if c in df_m.columns), "cantidad_casos")
    if case_col in df_m.columns:
        df_m[case_col] = pd.to_numeric(df_m[case_col], errors="coerce").fillna(0).clip(lower=0)
    
    # Normalizar provincia
    def fix_prov_name_simple(p):
        if pd.isna(p): return p
        p = str(p).strip().upper()
        if p in {"CABA","CIUDAD AUTONOMA BUENOS AIRES","CAPITAL FEDERAL","CIUDAD AUTONOMA DE BUENOS AIRES"}:
            return "CIUDAD AUTONOMA DE BUENOS AIRES"
        repl = str.maketrans("ÁÉÍÓÚÑ", "AEIOUN")
        return p.translate(repl)
    
    if "provincia_nombre" in df_m.columns:
        df_m["provincia_nombre"] = df_m["provincia_nombre"].apply(fix_prov_name_simple)
    
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
    
    if "provincia_nombre" in df_m.columns:
        df_m["clima_region"] = df_m["provincia_nombre"].map(PROVINCIA_A_CLIMA).fillna("TEMPLADO")
    else:
        df_m["clima_region"] = "TEMPLADO"
    
    # Crear fecha_semana si no existe
    if "fecha_semana" not in df_m.columns and {"anio","semana_epidemiologica"}.issubset(df_m.columns):
        def safe_fecha_semana(r):
            try:
                if pd.notna(r["anio"]) and pd.notna(r["semana_epidemiologica"]):
                    anio = int(r["anio"])
                    semana = int(r["semana_epidemiologica"])
                    # Limitar semana a máximo 52 (algunos años tienen 53, pero la mayoría 52)
                    if semana > 52:
                        semana = 52
                    return pd.to_datetime(date.fromisocalendar(anio, semana, 1))
                return pd.NaT
            except:
                return pd.NaT
        df_m["fecha_semana"] = df_m.apply(safe_fecha_semana, axis=1)
    
    df_m["fecha_semana"] = pd.to_datetime(df_m["fecha_semana"], errors="coerce")
    
    # Filtrar meses enero-junio (como en TP3)
    if "mes" not in df_m.columns:
        df_m["mes"] = df_m["fecha_semana"].dt.month.astype("Int64")
    df_m = df_m[df_m["mes"].between(1, 6)].copy()
    
    # Crear promedios semanales de clima (temp, hum, prec)
    dias = ["_L","_M","_X","_J","_V","_S","_D"]
    for base in ["temp","hum","prec"]:
        cols = [c for c in df_m.columns if c.lower().startswith(base + "_") and any(c.endswith(d) for d in dias)]
        if cols:
            df_m[f"{base}_sem_prom"] = df_m[cols].mean(axis=1)
    
    # Agrupar por provincia/departamento/semana/grupo_edad (como en TP3)
    group_cols = ["provincia_nombre", "departamento_nombre", "anio", "semana_epidemiologica", "fecha_semana", "clima_region"]
    agg_cols = ["temp_sem_prom", "hum_sem_prom", "prec_sem_prom", "densidad"]
    
    # Solo incluir columnas que existan
    group_cols = [c for c in group_cols if c in df_m.columns]
    agg_cols = [c for c in agg_cols if c in df_m.columns]
    
    # Agregación
    agg_dict = {case_col: "sum"}
    for col in agg_cols:
        agg_dict[col] = "mean"
    
    df_metrics = df_m.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # --- CÁLCULO DE KPIs ---
    total_casos = df_metrics[case_col].sum()
    
    # Casos promedio por semana
    casos_por_semana = df_metrics.groupby(["anio","semana_epidemiologica"], as_index=False)[case_col].sum()
    prom_casos_semana = casos_por_semana[case_col].mean()
    
    # Región más afectada
    casos_por_region = df_metrics.groupby("clima_region")[case_col].sum().sort_values(ascending=False)
    region_mas_afectada = casos_por_region.index[0] if len(casos_por_region) > 0 else "N/A"
    casos_region_max = casos_por_region.iloc[0] if len(casos_por_region) > 0 else 0
    
    # Temperatura y humedad promedio (de datos agregados)
    temp_prom = df_metrics["temp_sem_prom"].mean() if "temp_sem_prom" in df_metrics.columns else float('nan')
    hum_prom = df_metrics["hum_sem_prom"].mean() if "hum_sem_prom" in df_metrics.columns else float('nan')
    
    # Provincia más afectada
    casos_por_prov = df_metrics.groupby("provincia_nombre")[case_col].sum().sort_values(ascending=False)
    prov_mas_afectada = casos_por_prov.index[0] if len(casos_por_prov) > 0 else "N/A"
    casos_prov_max = casos_por_prov.iloc[0] if len(casos_por_prov) > 0 else 0
    
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
    # =====================
    # Análisis Feb-Mar-Abr 2024-2025
    # =====================
    st.subheader("🗓️ FMA (Febrero-Marzo-Abril) 2024-2025")
    if "fecha_semana" in df_metrics.columns:
        fma_mask = (df_metrics["fecha_semana"].dt.year.isin([2024, 2025]) &
                    df_metrics["fecha_semana"].dt.month.isin([2,3,4]))
        df_fma = df_metrics[fma_mask].copy()
        if not df_fma.empty:
            semanas_fma = df_fma.groupby(["anio","semana_epidemiologica"], as_index=False)[case_col].sum()
            prom_sem_fma = semanas_fma[case_col].mean()
            # Semana pico (máximo casos)
            pico_row = semanas_fma.loc[semanas_fma[case_col].idxmax()]
            pico_anio = pico_row["anio"]
            pico_sem = pico_row["semana_epidemiologica"]
            df_pico = df_fma[(df_fma["anio"]==pico_anio) & (df_fma["semana_epidemiologica"]==pico_sem)]
            temp_pico = df_pico["temp_sem_prom"].mean() if "temp_sem_prom" in df_pico.columns else float('nan')
            hum_pico = df_pico["hum_sem_prom"].mean() if "hum_sem_prom" in df_pico.columns else float('nan')
            prec_pico = df_pico["prec_sem_prom"].mean() if "prec_sem_prom" in df_pico.columns else float('nan')
            prec_prom_fma = df_fma["prec_sem_prom"].mean() if "prec_sem_prom" in df_fma.columns else float('nan')
            # Densidad promedio en todo el período FMA y en la semana pico
            dens_prom_fma = df_fma["densidad"].mean() if "densidad" in df_fma.columns else float('nan')
            dens_pico = df_pico["densidad"].mean() if "densidad" in df_pico.columns else float('nan')

            col_fma1, col_fma2, col_fma3 = st.columns(3)
            with col_fma1:
                st.metric("Promedio Semanal FMA", f"{prom_sem_fma:.1f}")
            with col_fma2:
                st.metric("Temp Semana Pico", f"{temp_pico:.1f}°C", help=f"Semana {int(pico_sem)} - {int(pico_anio)}")
            with col_fma3:
                st.metric("Hum Semana Pico", f"{hum_pico:.1f}%", help=f"Semana {int(pico_sem)} - {int(pico_anio)}")
            col_fma4, col_fma5, col_fma6 = st.columns(3)
            with col_fma4:
                st.metric("Densidad Promedio FMA", f"{dens_prom_fma:.1f}" if not pd.isna(dens_prom_fma) else "N/D")
            with col_fma5:
                st.metric("Densidad Semana Pico", f"{dens_pico:.1f}" if not pd.isna(dens_pico) else "N/D", help=f"Semana {int(pico_sem)} - {int(pico_anio)}")
            with col_fma6:
                st.metric("Prec Semana Pico", f"{prec_pico:.1f} mm" if not pd.isna(prec_pico) else "N/D", help=f"Semana {int(pico_sem)} - {int(pico_anio)}")
         

            # ...existing code...
        else:
            st.info("No hay datos para Febrero-Marzo-Abril 2024-2025 en el archivo.")
    else:
        st.info("No se puede calcular FMA: falta 'fecha_semana'.")
    st.markdown("---")
    
    # --- Gráficos de contexto ---

    st.subheader("📈 Estacionalidad de Casos por Mes y Año")
    # Heatmap de casos promedio por mes y año
    if "fecha_semana" in df_metrics.columns:
        df_metrics["mes"] = df_metrics["fecha_semana"].dt.month
        df_metrics["anio"] = df_metrics["fecha_semana"].dt.year
        df_mes_anio = df_metrics.groupby(["anio","mes"], as_index=False)[case_col].mean()
        heatmap = alt.Chart(df_mes_anio).mark_rect().encode(
            x=alt.X("anio:O", title="Año"),
            y=alt.Y("mes:O", title="Mes"),
            color=alt.Color(f"{case_col}:Q", title="Casos promedio", scale=alt.Scale(scheme="reds")),
            tooltip=["anio","mes",alt.Tooltip(f"{case_col}:Q", format=".1f", title="Casos promedio")]
        ).properties(title="Estacionalidad: concentración de casos por mes y año", width=500, height=300)
        st.altair_chart(heatmap, use_container_width=True)

        # Análisis textual de los meses con más casos
        top_meses = df_mes_anio.groupby("mes")[case_col].mean().sort_values(ascending=False).head(3)
        meses_map = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
        st.markdown(f"**Meses con mayor promedio de casos:** {'; '.join([meses_map[m] for m in top_meses.index])}")
        st.caption(f"Promedio de casos por mes: {', '.join([f'{meses_map[m]}: {int(v):,}' for m,v in top_meses.items()])}")

    st.markdown("---")
    st.subheader("📊 Impacto de Variables Climáticas (Global)")
    # Correlación global entre casos y clima
    clima_vars = ["temp_sem_prom", "hum_sem_prom", "prec_sem_prom"]
    df_clima = df_metrics.dropna(subset=clima_vars + [case_col])
    corrs = []
    for v in clima_vars:
        corr = df_clima[v].corr(df_clima[case_col])
        corrs.append({"variable": v, "corr": corr})
    corr_df = pd.DataFrame(corrs)
    corr_df["variable"] = corr_df["variable"].map({"temp_sem_prom":"Temperatura","hum_sem_prom":"Humedad","prec_sem_prom":"Precipitación"})
    bar_corr = alt.Chart(corr_df).mark_bar().encode(
        x=alt.X("variable:N", title="Variable climática"),
        y=alt.Y("corr:Q", title="Correlación con casos"),
        color=alt.Color("variable:N", legend=None),
        tooltip=["variable:N",alt.Tooltip("corr:Q", format=".2f")]
    ).properties(title="Correlación global entre variables climáticas y casos de dengue", width=500, height=300)
    st.altair_chart(bar_corr, use_container_width=True)

    # Análisis textual
    var_max = corr_df.loc[corr_df["corr"].abs().idxmax()]
    st.markdown(f"**La variable con mayor impacto sobre los casos es:** <span style='color:orange'><b>{var_max['variable']}</b></span> (correlación = {var_max['corr']:.2f})", unsafe_allow_html=True)


    # Impacto global: densidad y lags (similar a impacto climático)
    st.subheader("📊 Impacto de Densidad y Lags (Global)")
    keys = [k for k in ["provincia_nombre","departamento_nombre"] if k in df_metrics.columns]
    sort_keys = (keys + ["fecha_semana"]) if "fecha_semana" in df_metrics.columns else (keys + ["anio","semana_epidemiologica"]) 
    df_metrics = df_metrics.sort_values(sort_keys).copy()
    for n in [1,2,4]:
        coln = f"lag{n}"
        if coln not in df_metrics.columns:
            if keys:
                df_metrics[coln] = df_metrics.groupby(keys)[case_col].shift(n)
            else:
                df_metrics[coln] = df_metrics[case_col].shift(n)
    # Preparar correlaciones
    for c in ["densidad","lag1","lag2","lag4"]:
        if c in df_metrics.columns and df_metrics[c].dtype == "O":
            df_metrics[c] = pd.to_numeric(df_metrics[c], errors="coerce")
    impact_vars = [v for v in ["densidad","lag1","lag2","lag4"] if v in df_metrics.columns]
    corr_rows = []
    for v in impact_vars:
        sub = df_metrics[[v, case_col]].dropna()
        if not sub.empty:
            corr_rows.append({"variable": v, "corr": sub[v].corr(sub[case_col])})
    if corr_rows:
        corr_df2 = pd.DataFrame(corr_rows)
        label_map = {"densidad":"Densidad","lag1":"Lag 1 semana","lag2":"Lag 2 semanas","lag4":"Lag 4 semanas"}
        corr_df2["variable"] = corr_df2["variable"].map(label_map)
        bar_imp = alt.Chart(corr_df2).mark_bar().encode(
            x=alt.X("variable:N", title="Variable"),
            y=alt.Y("corr:Q", title="Correlación con casos"),
            color=alt.Color("variable:N", legend=None),
            tooltip=["variable:N", alt.Tooltip("corr:Q", format=".2f")]
        ).properties(title="Impacto de densidad y lags (correlación global)", width=500, height=280)
        st.altair_chart(bar_imp, use_container_width=True)
    else:
        st.info("No hay suficientes datos para calcular correlaciones de densidad y lags.")
    st.markdown("---")
    st.caption("*Métricas calculadas en tiempo real desde `info/dengue_enriched_final.xlsx`*")

with tab4:
    st.header("📈 Dashboards")
    st.markdown("---")
    st.subheader("Huella por región (z-score por variable)")

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
        try:
            y = int(year)
            w = int(week)
            # Limitar a máximo 52 semanas
            if w > 52:
                w = 52
            return pd.to_datetime(date.fromisocalendar(y, w, 1))
        except:
            return pd.NaT
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
