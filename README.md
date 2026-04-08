# 销售预测系统（真实可用版）

本项目提供可落地的销售预测系统代码，包含：

- **数据接入**：内部 CSV / API 数据接入、外部 API（示例：天气）接入。
- **特征工程**：时间特征、滞后特征、滚动统计、外部因子融合。
- **预测引擎**：基于模型注册中心执行训练、预测与批量推理。
- **模型管理**：按名称注册模型、训练后持久化、版本元数据。
- **结果评估**：MAE / RMSE / MAPE / WAPE。
- **可视化看板**：Streamlit + Plotly 展示预测结果、误差、趋势。
- **协同管理**：支持预测任务审批流、人工干预记录、发布前审核。
- **消息通知**：支持控制台/Webhook 通知（可扩展企业微信、钉钉、邮件）。
- **报告导出与分享**：预测评估结果支持导出 Excel、PDF 并按角色分享。

## 业务规则支持

1. 支持多维度预测：产品、客户、区域、渠道。  
2. 预测时间粒度可配置：日/周/月/季。  
3. 结果包含：预测值、置信区间、同比/环比、趋势结论。  
4. 支持与历史实际值/销售目标对比，自动计算准确率、MAE、MAPE。  
5. 支持报告导出（Excel、PDF）与角色分享。  

## 模型与版本能力

- 内置模型类型：ARIMA、Prophet、Random Forest、XGBoost、LightGBM、LSTM、融合模型（weighted/fusion）。
- 支持按场景自动推荐模型：常规品、爆品、季节品、新品上市。
- 支持可配置参数：历史数据周期（6/12/24 月）、预测周期（1-12 月）、置信区间（默认 95%）。
- 支持配置内外部影响因子并量化权重（如原材料价格、市场活动、政策新规）。
- 自动计算评估指标：Accuracy、MAE、MSE、RMSE、MAPE、WAPE。
- 支持模型版本管理：记录版本、指标、参数，并支持回滚到历史版本。

## 预测复盘能力

- 复盘周期可配置：每周/每月/每季度，支持手动触发复盘。
- 复盘指标内置：Accuracy、MAE、MSE、偏差率，并支持注册自定义指标。
- 偏差成因分析：支持按产品、区域、时间维度拆解，并区分系统性偏差与偶然性偏差。
- 复盘报告内容：偏差明细表格、成因分析（文本图表化）、优化建议。
- 支持复盘报告导出与分享，并记录优化措施落实情况。

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
