import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# ----------------------------
# 📌 기본 CSV 파일 경로 설정
DEFAULT_FILE = "기온데이터(utf-8).csv"
# ----------------------------

st.set_page_config(page_title="기온 추세 분석", layout="wide")
st.title("🌡️ 연도별 기온 추세 분석 대시보드")

# ----------------------------
# 📂 CSV 파일 불러오기 (업로드 우선, 없으면 기본 파일)
# ----------------------------
uploaded_file = st.file_uploader("기온 데이터 파일 업로드 (선택)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 업로드한 CSV 파일이 성공적으로 로드되었습니다.")
elif os.path.exists(DEFAULT_FILE):
    df = pd.read_csv(DEFAULT_FILE)
    st.info("ℹ️ 기본 파일 '기온데이터(utf-8).csv'이 사용되었습니다.")
else:
    st.error("❌ 기본 CSV 파일이 없고 업로드된 파일도 없습니다.")
    st.stop()

# ----------------------------
# ⏳ 데이터 전처리
# ----------------------------
try:
    df["일시"] = pd.to_datetime(df["일시"])
    df["연도"] = df["일시"].dt.year

    for col in ["평균기온(°C)", "최저기온(°C)", "최고기온(°C)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    yearly = df.groupby("연도")[["평균기온(°C)", "최저기온(°C)", "최고기온(°C)"]].mean().reset_index()
except Exception as e:
    st.error(f"❌ 데이터 처리 중 오류 발생: {e}")
    st.stop()

# ----------------------------
# 📊 연도별 기온 추세 시각화 (Plotly)
# ----------------------------
st.subheader("1️⃣ 연도별 평균/최저/최고 기온 추세선")

fig = go.Figure()
fig.add_trace(go.Scatter(x=yearly["연도"], y=yearly["평균기온(°C)"],
                         mode='lines+markers', name="평균기온(°C)"))
fig.add_trace(go.Scatter(x=yearly["연도"], y=yearly["최저기온(°C)"],
                         mode='lines+markers', name="최저기온(°C)"))
fig.add_trace(go.Scatter(x=yearly["연도"], y=yearly["최고기온(°C)"],
                         mode='lines+markers', name="최고기온(°C)"))

fig.update_layout(title="연도별 기온 추세",
                  xaxis_title="연도",
                  yaxis_title="기온 (°C)",
                  hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# ⚙️ 추세선 생성 기능
# ----------------------------
st.subheader("2️⃣ 연도 범위 설정 및 추세선 예측")

year_min, year_max = int(df["연도"].min()), int(df["연도"].max())

col1, col2 = st.columns(2)
with col1:
    input_range = st.slider("📌 입력 데이터 연도 범위", year_min, year_max, (year_min, year_max))
with col2:
    pred_range = st.slider("🔮 예측 연도 범위", year_max + 1, year_max + 5, (year_max + 1, year_max + 3))

if st.button("📈 추세선 예측"):
    input_df = yearly[(yearly["연도"] >= input_range[0]) & (yearly["연도"] <= input_range[1])]
    pred_years = np.array(range(pred_range[0], pred_range[1] + 1)).reshape(-1, 1)

    fig2 = go.Figure()
    colors = {"평균기온(°C)": "blue", "최저기온(°C)": "green", "최고기온(°C)": "red"}

    for col in ["평균기온(°C)", "최저기온(°C)", "최고기온(°C)"]:
        model = LinearRegression()
        model.fit(input_df[["연도"]], input_df[col])
        pred_values = model.predict(pred_years)

        # 실제 값
        fig2.add_trace(go.Scatter(
            x=input_df["연도"], y=input_df[col],
            mode="lines+markers", name=f"{col} (입력)", line=dict(color=colors[col])
        ))

        # 추세선
        fig2.add_trace(go.Scatter(
            x=pred_years.flatten(), y=pred_values,
            mode="lines+markers", name=f"{col} 예측", line=dict(dash="dash", color=colors[col])
        ))

    fig2.update_layout(title="📈 기온 추세선 예측 결과",
                       xaxis_title="연도", yaxis_title="기온 (°C)",
                       hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)
