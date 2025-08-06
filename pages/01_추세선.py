import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import os

# ----------------------------
# 📁 기본 파일 경로 설정
DEFAULT_FILE = "기온데이터(utf-8).csv"
# ----------------------------

st.set_page_config(page_title="기온 추세 분석", layout="wide")
st.title("🌡️ 연도별 및 월별 기온 추세 분석 대시보드")

# ----------------------------
# 📂 파일 업로드 or 기본 사용
uploaded_file = st.file_uploader("기온 데이터 CSV 업로드 (선택)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 업로드한 파일을 사용 중입니다.")
elif os.path.exists(DEFAULT_FILE):
    df = pd.read_csv(DEFAULT_FILE)
    st.info("ℹ️ 기본 파일 '기온데이터(utf-8).csv'을 사용 중입니다.")
else:
    st.error("❌ 사용할 수 있는 파일이 없습니다.")
    st.stop()

# ----------------------------
# ⏳ 전처리
try:
    df["날짜"] = df["날짜"].astype(str).str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["날짜"])  # 유효한 날짜만
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month

    for col in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
except Exception as e:
    st.error(f"❌ 전처리 중 오류 발생: {e}")
    st.stop()

# ----------------------------
# 📊 연도별 시각화
st.subheader("1️⃣ 연도별 기온 추세")
yearly = df.groupby("연도")[["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]].mean().reset_index()

fig_year = go.Figure()
fig_year.add_trace(go.Scatter(x=yearly["연도"], y=yearly["평균기온(℃)"], mode='lines+markers', name="평균기온"))
fig_year.add_trace(go.Scatter(x=yearly["연도"], y=yearly["최저기온(℃)"], mode='lines+markers', name="최저기온"))
fig_year.add_trace(go.Scatter(x=yearly["연도"], y=yearly["최고기온(℃)"], mode='lines+markers', name="최고기온"))

fig_year.update_layout(title="연도별 기온 추세",
                       xaxis_title="연도", yaxis_title="기온 (℃)",
                       hovermode="x unified")
st.plotly_chart(fig_year, use_container_width=True)

# ----------------------------
# 📅 월별 시각화
st.subheader("2️⃣ 월별 평균 기온 (전체 연도 기준)")
monthly = df.groupby("월")[["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]].mean().reset_index()

fig_month = go.Figure()
fig_month.add_trace(go.Scatter(x=monthly["월"], y=monthly["평균기온(℃)"], mode='lines+markers', name="평균기온"))
fig_month.add_trace(go.Scatter(x=monthly["월"], y=monthly["최저기온(℃)"], mode='lines+markers', name="최저기온"))
fig_month.add_trace(go.Scatter(x=monthly["월"], y=monthly["최고기온(℃)"], mode='lines+markers', name="최고기온"))

fig_month.update_layout(title="월별 평균 기온 (전체 연도 기준)",
                        xaxis=dict(tickmode='linear'),  # 1~12월
                        xaxis_title="월", yaxis_title="기온 (℃)",
                        hovermode="x unified")
st.plotly_chart(fig_month, use_container_width=True)

# ----------------------------
# 🔮 추세선 예측
st.subheader("3️⃣ 연도별 추세선 예측")

year_min, year_max = int(df["연도"].min()), int(df["연도"].max())

col1, col2 = st.columns(2)
with col1:
    input_range = st.slider("📌 입력 데이터 연도 범위", year_min, year_max, (year_min, year_max))
with col2:
    pred_range = st.slider("🔮 예측 연도 범위", year_max + 1, year_max + 5, (year_max + 1, year_max + 3))

if st.button("📈 추세선 예측하기"):
    input_df = yearly[(yearly["연도"] >= input_range[0]) & (yearly["연도"] <= input_range[1])]
    pred_years = np.array(range(pred_range[0], pred_range[1] + 1)).reshape(-1, 1)

    fig_pred = go.Figure()
    colors = {"평균기온(℃)": "blue", "최저기온(℃)": "green", "최고기온(℃)": "red"}

    for col in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        df_valid = input_df[["연도", col]].dropna()
        if df_valid.empty:
            st.warning(f"⚠️ {col}에 사용할 수 있는 데이터가 없습니다.")
            continue

        model = LinearRegression()
        model.fit(df_valid[["연도"]], df_valid[col])
        pred_values = model.predict(pred_years)

        fig_pred.add_trace(go.Scatter(
            x=df_valid["연도"], y=df_valid[col],
            mode="lines+markers", name=f"{col} (입력)", line=dict(color=colors[col])
        ))

        fig_pred.add_trace(go.Scatter(
            x=pred_years.flatten(), y=pred_values,
            mode="lines+markers", name=f"{col} 예측", line=dict(dash="dash", color=colors[col])
        ))

    fig_pred.update_layout(title="📈 기온 추세선 예측 결과",
                           xaxis_title="연도", yaxis_title="기온 (℃)",
                           hovermode="x unified")
    st.plotly_chart(fig_pred, use_container_width=True)
