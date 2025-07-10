import streamlit as st
import pandas as pd
import datetime
import os
import plotly.express as px

st.set_page_config(page_title="어제 기온 vs 역대 기온", layout="centered")
st.title("📈 어제는 얼마나 더웠을까?")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요 (기온 데이터, CP949 인코딩)", type="csv")

# 업로드된 파일이 없으면 기본 파일 사용
if not uploaded_file:
    for fname in os.listdir("."):
        if fname.startswith("ta") and fname.endswith(".csv"):
            uploaded_file = fname
            break

if uploaded_file:
    try:
        file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file

        df = pd.read_csv(file_path, encoding="cp949", skiprows=7)
        df["날짜"] = df["날짜"].str.strip()
        df["날짜"] = pd.to_datetime(df["날짜"], format="%Y-%m-%d")

        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        st.subheader(f"🔍 분석 대상 날짜: {yesterday.strftime('%Y-%m-%d')}")

        df_yesterday = df[df["날짜"] == pd.to_datetime(yesterday)]
        if df_yesterday.empty:
            st.warning("해당 날짜의 데이터가 없습니다.")
        else:
            same_day_df = df[df["날짜"].dt.strftime("%m-%d") == yesterday.strftime("%m-%d")]

            highest_temp_yesterday = df_yesterday["최고기온(℃)"].values[0]
            highest_ranks = same_day_df.sort_values("최고기온(℃)", ascending=False).reset_index(drop=True)
            highest_rank = highest_ranks[highest_ranks["날짜"] == pd.to_datetime(yesterday)].index[0] + 1
            highest_percentile = 100 * (highest_rank / len(highest_ranks))

            lowest_temp_yesterday = df_yesterday["최저기온(℃)"].values[0]
            lowest_ranks = same_day_df.sort_values("최저기온(℃)").reset_index(drop=True)
            lowest_rank = lowest_ranks[lowest_ranks["날짜"] == pd.to_datetime(yesterday)].index[0] + 1
            lowest_percentile = 100 * (lowest_rank / len(lowest_ranks))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌡️ 어제 최고기온", f"{highest_temp_yesterday}℃", f"상위 {highest_percentile:.1f}%")
                st.info(f"역대 7월 {yesterday.day}일 중 **{highest_rank}위**")
            with col2:
                st.metric("🌙 어제 최저기온", f"{lowest_temp_yesterday}℃", f"상위 {lowest_percentile:.1f}%")
                st.info(f"역대 7월 {yesterday.day}일 중 **{lowest_rank}위**")

            st.markdown("---")
            st.subheader("🔥 역대 동일 날짜 중 가장 더웠던 날 Top 5")
            st.dataframe(highest_ranks.head(5).reset_index(drop=True))

            st.subheader("📊 최고기온 추이 (Plotly)")
            fig_high = px.line(same_day_df.sort_values("날짜"), x="날짜", y="최고기온(℃)",
                               title="역대 7월 {}일 최고기온 추이".format(yesterday.day))
            fig_high.add_scatter(x=[yesterday], y=[highest_temp_yesterday], mode='markers+text',
                                 name='어제', marker=dict(size=12, color='red'),
                                 text=["어제"], textposition="top center")
            st.plotly_chart(fig_high)

            st.markdown("---")
            st.subheader("❄️ 역대 동일 날짜 중 가장 추웠던 날 Top 5")
            st.dataframe(lowest_ranks.head(5).reset_index(drop=True))

            st.subheader("📊 최저기온 추이 (Plotly)")
            fig_low = px.line(same_day_df.sort_values("날짜"), x="날짜", y="최저기온(℃)",
                              title="역대 7월 {}일 최저기온 추이".format(yesterday.day))
            fig_low.add_scatter(x=[yesterday], y=[lowest_temp_yesterday], mode='markers+text',
                                name='어제', marker=dict(size=12, color='blue'),
                                text=["어제"], textposition="top center")
            st.plotly_chart(fig_low)

            st.markdown("---")
            st.subheader("📅 최근 1주일간 평균 기온 분석")
            one_week_ago = pd.to_datetime(today) - pd.Timedelta(days=7)
            last_week_df = df[(df["날짜"] >= one_week_ago) & (df["날짜"] < pd.to_datetime(today))]

            avg_high = last_week_df["최고기온(℃)"].mean()
            avg_low = last_week_df["최저기온(℃)"].mean()
            avg_avg = last_week_df["평균기온(℃)"].mean()

            st.write(f"최근 7일간 평균 최고기온: **{avg_high:.2f}℃**")
            st.write(f"최근 7일간 평균 최저기온: **{avg_low:.2f}℃**")
            st.write(f"최근 7일간 평균기온: **{avg_avg:.2f}℃**")

            fig_week = px.line(last_week_df.sort_values("날짜"), x="날짜", y=["최고기온(℃)", "평균기온(℃)", "최저기온(℃)"],
                               title="최근 1주일간 기온 변화 추이")
            st.plotly_chart(fig_week)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
