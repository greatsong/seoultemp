# 어제 기온 vs 역대 기온  ─────────────────────────────────────────
# (c) 2025 Example – 자유롭게 수정·재배포 가능
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import datetime
import os
import plotly.express as px
#import koreanize_matplotlib   # 한글 폰트 자동 설정 (그래프·표시용)

# ────────────────────────────────────────────────────────────────
# 1. 페이지 설정
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="어제 기온 vs 역대 기온",
    layout="centered",
    page_icon="📈",
)
st.title("📈 어제는 얼마나 더웠을까?")

# ────────────────────────────────────────────────────────────────
# 2. 데이터 로더
# ────────────────────────────────────────────────────────────────
def load_temperature_csv(path_or_file, skiprows: int = 7) -> pd.DataFrame:
    """CP949 → UTF-8-SIG 순으로 시도하여 CSV 로드"""
    for enc in ("cp949", "utf-8-sig"):
        try:
            df = pd.read_csv(path_or_file, encoding=enc, skiprows=skiprows)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("지원되지 않는 인코딩입니다.")

    # 날짜 전처리
    if "날짜" not in df.columns:
        raise ValueError("CSV에 '날짜' 열이 없습니다.")
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), format="%Y-%m-%d")
    return df


# ────────────────────────────────────────────────────────────────
# 3. CSV 업로드 또는 기본 파일 사용
# ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "CSV 파일을 업로드하세요 (CP949 또는 UTF-8, 7행 헤더 제외)",
    type="csv",
)

if uploaded_file is None:
    # 업로드가 없으면 기본 파일 찾기
    default_file = next(
        (fname for fname in os.listdir(".") if fname.startswith("ta") and fname.endswith(".csv")),
        None,
    )
    if default_file:
        uploaded_file = default_file
        st.info(f"기본 파일 **{default_file}** 을(를) 사용합니다.")
    else:
        st.warning("CSV 파일을 업로드하거나 작업 폴더에 'ta*.csv' 파일을 두세요.")
        st.stop()

# 데이터 불러오기
try:
    df = load_temperature_csv(uploaded_file)
except Exception as e:
    st.error(f"파일을 읽는 중 오류 발생: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────
# 4. 분석 대상 날짜(어제) 설정
# ────────────────────────────────────────────────────────────────
today      = datetime.date.today()
yesterday  = today - datetime.timedelta(days=1)
yesterday_dt = pd.to_datetime(yesterday)

st.subheader(f"🔍 분석 대상 날짜: {yesterday:%Y-%m-%d}")

df_yest = df[df["날짜"] == yesterday_dt]
if df_yest.empty:
    st.warning("데이터에 어제 날짜가 없습니다. CSV를 확인해 주세요.")
    st.stop()

# ────────────────────────────────────────────────────────────────
# 5. 연도 범위 슬라이더
# ────────────────────────────────────────────────────────────────
year_min, year_max = df["날짜"].dt.year.min(), df["날짜"].dt.year.max()
sel_years = st.slider(
    "비교할 연도 범위",
    min_value=int(year_min),
    max_value=int(year_max),
    value=(int(year_min), int(year_max)),
)

# 어제와 같은 MM-DD를 가진 행만 추출
same_day_df = df[
    (df["날짜"].dt.strftime("%m-%d") == yesterday.strftime("%m-%d"))
    & (df["날짜"].dt.year.between(*sel_years))
]

# ────────────────────────────────────────────────────────────────
# 6. 최고·최저 기온 랭킹 계산
# ────────────────────────────────────────────────────────────────
high_temp = df_yest["최고기온(℃)"].iloc[0]
low_temp  = df_yest["최저기온(℃)"].iloc[0]

rank_high_df = same_day_df.sort_values("최고기온(℃)", ascending=False).reset_index(drop=True)
rank_low_df  = same_day_df.sort_values("최저기온(℃)").reset_index(drop=True)

rank_high = rank_high_df[rank_high_df["날짜"] == yesterday_dt].index[0] + 1  # 1-based
rank_low  = rank_low_df[rank_low_df["날짜"] == yesterday_dt].index[0] + 1

pct_high = 100 * (rank_high - 1) / len(rank_high_df)  # 1위가 0 %
pct_low  = 100 * (rank_low  - 1) / len(rank_low_df)

rec_high = rank_high_df.iloc[0]
rec_low  = rank_low_df.iloc[0]

# ────────────────────────────────────────────────────────────────
# 7. 결과 표시
# ────────────────────────────────────────────────────────────────
st.markdown("### 🏆 역대 기록 요약")
st.write(
    f"📈 **역대 최고기온**: {rec_high['최고기온(℃)']}℃ "
    f"({rec_high['날짜'].date()}) → 어제보다 "
    f"{rec_high['최고기온(℃)'] - high_temp:+.1f}℃"
)
st.write(
    f"❄️ **역대 최저기온**: {rec_low['최저기온(℃)']}℃ "
    f"({rec_low['날짜'].date()}) → 어제보다 "
    f"{rec_low['최저기온(℃)'] - low_temp:+.1f}℃"
)

c1, c2 = st.columns(2)
c1.metric("🌡️ 어제 최고기온", f"{high_temp}℃", f"상위 {pct_high:.1f}%")
c2.metric("🌙 어제 최저기온", f"{low_temp}℃",  f"상위 {pct_low:.1f}%")

# ────────────────────────────────────────────────────────────────
# 8. Top 5 테이블 & 추이 그래프
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔥 역대 동일 날짜 중 가장 더웠던 날 Top 5")
st.dataframe(rank_high_df.head(5).reset_index(drop=True))

fig_high = px.line(
    same_day_df.sort_values("날짜"),
    x="날짜", y="최고기온(℃)",
    title=f"역대 {yesterday:%m월 %d일} 최고기온 추이",
)
fig_high.add_scatter(
    x=[yesterday_dt], y=[high_temp],
    mode="markers+text",
    name="어제", marker=dict(size=12, color="red"),
    text=["어제"], textposition="top center",
)
st.plotly_chart(fig_high, use_container_width=True)

st.markdown("---")
st.subheader("❄️ 역대 동일 날짜 중 가장 추웠던 날 Top 5")
st.dataframe(rank_low_df.head(5).reset_index(drop=True))

fig_low = px.line(
    same_day_df.sort_values("날짜"),
    x="날짜", y="최저기온(℃)",
    title=f"역대 {yesterday:%m월 %d일} 최저기온 추이",
)
fig_low.add_scatter(
    x=[yesterday_dt], y=[low_temp],
    mode="markers+text",
    name="어제", marker=dict(size=12, color="blue"),
    text=["어제"], textposition="top center",
)
st.plotly_chart(fig_low, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# 9. 최근 N일 평균 vs 역대 동일 기간 평균
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 최근 기간 평균 기온 분석")

day_range = st.slider("비교할 최근 일 수", 3, 30, 7)
start_day = pd.to_datetime(today) - pd.Timedelta(days=day_range)
recent_df = df[(df["날짜"] >= start_day) & (df["날짜"] < pd.to_datetime(today))]

avg_high = recent_df["최고기온(℃)"].mean()
avg_low  = recent_df["최저기온(℃)"].mean()
avg_mean = recent_df["평균기온(℃)"].mean()

# 동일 기간(과거 연도들의 같은 MM-DD) 평균
period_days = [(today - datetime.timedelta(days=i)).strftime("%m-%d") for i in range(1, day_range + 1)]
hist_df = df[df["날짜"].dt.strftime("%m-%d").isin(period_days)]
hist_avg = (
    hist_df
    .groupby(hist_df["날짜"].dt.strftime("%m-%d"))
    [["최고기온(℃)", "평균기온(℃)", "최저기온(℃)"]]
    .mean()
)
hist_high, hist_mean, hist_low = hist_avg.mean()

st.write(
    f"🌡️ 최근 {day_range}일 평균 **{avg_high:.2f}/{avg_mean:.2f}/{avg_low:.2f}℃** "
    f"(최고/평균/최저)\n"
    f"🗂️ 역대 동기간 평균 **{hist_high:.2f}/{hist_mean:.2f}/{hist_low:.2f}℃**"
)

# 백분위(평균기온 기준)
pct = 100 * (hist_avg["평균기온(℃)"] < avg_mean).sum() / len(hist_avg)
rank = len(hist_avg) - int(pct * len(hist_avg) / 100)
st.info(
    f"📈 최근 {day_range}일 평균기온은 역대 동일 기간 중 상위 **{100 - pct:.1f}%** "
    f"(전체 {len(hist_avg)}개 기간 중 {rank}위)입니다."
)

# ────────────────────────────────────────────────────────────────
# 10. 최고 vs 최저 Scatter 분포
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📍 최고기온 vs 최저기온 분포")

scatter_df = same_day_df.copy()
scatter_df["날짜(문자)"] = scatter_df["날짜"].dt.strftime("%Y-%m-%d")
scatter_df["어제"] = scatter_df["날짜"] == yesterday_dt

fig_scatter = px.scatter(
    scatter_df, x="최고기온(℃)", y="최저기온(℃)",
    color="어제", hover_name="날짜(문자)",
    title="역대 최고·최저 기온 분포(동일 날짜)",
    labels={"어제": "어제 여부"},
)
st.plotly_chart(fig_scatter, use_container_width=True)
