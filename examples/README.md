# 示例数据

- `simulation/`：确定性生成的 8 基因、80 细胞、4 时间窗模拟数据，附带生成机制的全局 GRN 标签。
- `real_a549_dex/`：GSE117089 A549 DEX 真实单细胞表达及公开 GRN 标签的小型子集；仅验证文件格式与命令行，不可用于性能结论。

每个目录中的 `expression.csv`、`time.csv`、`candidates.csv` 和 `labels.csv` 的字段含义见仓库根目录 `README.md`。
