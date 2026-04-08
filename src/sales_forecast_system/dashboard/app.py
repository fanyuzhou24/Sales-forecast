from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("销售预测可视化看板")

pred_file = st.sidebar.text_input("预测结果文件", value="artifacts/predictions.csv")
metric_file = st.sidebar.text_input("指标文件", value="artifacts/metrics.json")

pred_path = Path(pred_file)
metric_path = Path(metric_file)

if pred_path.exists():
    df = pd.read_csv(pred_path)
    st.subheader("预测结果总览")
    st.dataframe(df.head(30), use_container_width=True)

    if {"date", "item_id", "sales", "prediction"}.issubset(df.columns):
        item_options = sorted(df["item_id"].astype(str).unique())
        selected_item = st.selectbox("选择商品", item_options)
        item_df = df[df["item_id"].astype(str) == selected_item].copy()
        item_df["date"] = pd.to_datetime(item_df["date"])

        fig = px.line(item_df, x="date", y=["sales", "prediction"], title=f"商品 {selected_item} 销售 vs 预测")
        st.plotly_chart(fig, use_container_width=True)

        item_df["abs_error"] = (item_df["sales"] - item_df["prediction"]).abs()
        fig_err = px.bar(item_df, x="date", y="abs_error", title="绝对误差")
        st.plotly_chart(fig_err, use_container_width=True)
else:
    st.warning("未找到预测结果文件，请先执行训练/预测流水线。")

if metric_path.exists():
    st.subheader("评估指标")
    metrics = pd.read_json(metric_path, typ="series")
    st.json(metrics.to_dict())
