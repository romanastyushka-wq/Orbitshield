import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import time
import numpy as np
from db_handler import DatabaseHandler

# Настройка страницы
st.set_page_config(page_title="SolarShield AI Control", layout="wide", page_icon="☀️")
db = DatabaseHandler()

def get_data():
    df = db.get_recent_history(limit=50)
    # Превращаем строки времени из БД в реальные объекты datetime
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- SIDEBAR (Статистика) ---
with st.sidebar:
    st.header("SolarShield System")
    stats = db.get_stats()
    st.metric("Total Events Logged", stats['total_records'])
    st.metric("Historical Max Risk", f"{stats['max_risk_ever']:.4f}%") # Тоже увеличил точность
    st.markdown("---")
    st.caption("v2.8: Precision Mode")

st.title("🛰️ SolarShield: Integrated Satellite Protection")

# Контейнеры для организации интерфейса
header_placeholder = st.empty()
metrics_placeholder = st.empty()
main_layout = st.container()

while True:
    df = get_data()
    if not df.empty:
        last = df.iloc[-1]
        
        # --- БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ ДАННЫХ ---
        plasma_risk = last.get('risk_score', 0) if last.get('risk_score') is not None else 0.0
        vis_risk = last.get('visual_risk', 0) if last.get('visual_risk') is not None else 0.0
        speed = last.get('speed', 0) if last.get('speed') is not None else 0.0
        bz = last.get('bz', 0) if last.get('bz') is not None else 0.0
        kp = last.get('kp', 0) if last.get('kp') is not None else 0.0
        status_text = last.get('status', 'Monitoring')
        
        # Переменные вектора поля
        bx = last.get('bx', 0) if last.get('bx') is not None else 0.0
        by = last.get('by', 0) if last.get('by') is not None else 0.0
        
        # Форматируем время
        last_time_str = last['timestamp'].strftime('%H:%M:%S UTC')

        # --- БЛОК 0: ВРЕМЯ ---
        with header_placeholder.container():
             st.markdown(f"**⏱️ Last Data Update:** `{last_time_str}`")

        # --- БЛОК 1: МЕТРИКИ (ТОЧНЫЕ) ---
        with metrics_placeholder.container():
            m1, m2, m3, m4, m5 = st.columns(5)
            # ЗДЕСЬ ИЗМЕНЕНИЯ: .4f вместо .1f (4 знака после запятой)
            m1.metric("Plasma Status", f"{plasma_risk:.4f}%", delta=status_text)
            m2.metric("AI Vision Risk", f"{vis_risk:.4f}%")
            m3.metric("Wind Speed", f"{speed:.1f} km/s") # Скорость тоже чуть точнее
            m4.metric("Magnetic Bz", f"{bz:.2f} nT")
            m5.metric("Kp Index", f"{kp:.2f}")

        # --- БЛОК 2: ГРАФИКИ И ВИЗУАЛ ---
        with main_layout:
            col_graph, col_gauges, col_sun = st.columns([2, 1, 1])
            
            # 1. График истории
            with col_graph:
                st.subheader("Risk Evolution")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['risk_score'], 
                    name="Plasma Risk", fill='tozeroy', line=dict(color='red', width=2)
                ))
                fig_hist.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['visual_risk'], 
                    name="Vision Risk", line=dict(color='orange', width=2)
                ))
                fig_hist.update_layout(
                    height=450, template="plotly_dark", margin=dict(l=0,r=0,b=0,t=30),
                    xaxis=dict(tickformat="%H:%M:%S", title="Time"), # Добавил секунды на ось
                    yaxis=dict(range=[0, 100], title="Risk %"),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # 2. Спидометры
            with col_gauges:
                st.subheader("Physical Indicators")
                
                # Спидометр Риска
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = plasma_risk,
                    number = {'valueformat': '.2f'}, # Спидометр показывает 2 знака
                    title = {'text': "Plasma Anomaly", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "white"},
                        'steps': [
                            {'range': [0, 20], 'color': "#00cc96"},
                            {'range': [20, 50], 'color': "#ffa15a"},
                            {'range': [50, 100], 'color': "#ef553b"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': plasma_risk}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Радар (Bx/By)
                r_val = np.sqrt(bx**2 + by**2)
                theta_val = np.degrees(np.arctan2(by, bx))
                
                fig_polar = go.Figure(go.Barpolar(
                    r=[r_val], theta=[theta_val], width=[20], 
                    marker_color=["#ff4b4b" if r_val > 10 else "#0068c9"]
                ))
                fig_polar.update_layout(
                    height=200, 
                    title="IMF Direction",
                    polar=dict(radialaxis=dict(range=[0, 20], showticklabels=False)), 
                    margin=dict(l=30,r=30,t=30,b=20)
                )
                st.plotly_chart(fig_polar, use_container_width=True)

            # 3. Картинка Солнца (Визуал)
            with col_sun:
                st.subheader("SDO Visual Feed")
                if os.path.exists("latest_sun.jpg"):
                    st.image("latest_sun.jpg", use_container_width=True, caption=f"AI Scan: {vis_risk:.3f}% Risk")
                    
                    if vis_risk > 40:
                        st.error("🚨 VISUAL ANOMALY")
                    else:
                        st.success("Visual State: Quiet")
                else:
                    st.info("Waiting for image...")
                    st.caption("Ensure realtime_monitor.py is running")

    else:
        st.warning("Database empty. Start realtime_monitor.py first.")

    time.sleep(5)
    st.rerun()