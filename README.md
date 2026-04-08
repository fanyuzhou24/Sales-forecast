# 销售预测系统（真实可用版）

本项目提供可落地的销售预测系统代码，包含：

- **数据接入**：内部 CSV / API 数据接入、外部 API（示例：天气）接入。
- **特征工程**：时间特征、滞后特征、滚动统计、外部因子融合。
- **预测引擎**：基于模型注册中心执行训练、预测与批量推理。
- **模型管理**：按名称注册模型、训练后持久化、版本元数据。
- **结果评估**：MAE / RMSE / MAPE / WAPE。
- **可视化看板**：Streamlit + Plotly 展示预测结果、误差、趋势。

## 快速开始

```bash
pip install -e .
python -m sales_forecast_system.execution.runner   --train-data data/train.csv   --output-dir artifacts
```

启动 API：

```bash
uvicorn sales_forecast_system.app:app --reload --port 8080
```

启动看板：

```bash
streamlit run src/sales_forecast_system/dashboard/app.py
```
