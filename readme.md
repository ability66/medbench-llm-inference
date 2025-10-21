# MedBench LLM Inference

极简运行：

```bash
python generate.py
```

> 本仓库为 MedBench/医疗问答场景的 LLM 推理与评测脚本。按上面命令即可跑通默认流程。

---

## 环境要求
- Python ≥ 3.9（建议 3.10/3.11）
- 基础科学计算/通用库（以 `generate.py` 顶部 `import` 为准）

## 依赖安装（`requirements.txt`，由 **uv** 导出）
提供了 `requirements.txt`，你可以用 **pip** 或 **uv** 安装：

- 使用 pip：
  ```bash
  pip install -r requirements.txt
  ```
- 使用 uv（与 pip 兼容的子命令）：
  ```bash
  uv pip install -r requirements.txt
  ```

> 建议优先使用虚拟环境（`python -m venv .venv && source .venv/bin/activate`；Windows 用 `.venv\Scripts\activate`）。

## 快速开始
1. 克隆项目
   ```bash
   git clone https://github.com/ability66/medbench-llm-inference.git
   cd medbench-llm-inference
   ```
2. 创建虚拟环境
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
   ```
3. 安装依赖（任选其一）
   ```bash
   pip install -r requirements.txt
   # 或
   uv pip install -r requirements.txt
   ```
4. 运行评测
   ```bash
   python evaluate.py --compare_body_only
   ```

## 目录结构（概览）
```
IMCS-V2-MRG/       # 示例数据或相关资源（命名按仓库实际为准）
result/            # 运行输出目录（若脚本另有设置，以脚本为准）
generate.py        # 推理脚本
eval.py            # 评测脚本
```

## 基准结果（保留三位小数）
> 数据集规模与配置以实际运行为准；下表为提供的示例结果。

| model        | rouge-1 | rouge-2 | rouge-l | avg_rouge |   bleu | avg_chars |
| :----------- | ------: | ------: | ------: | --------: | -----: | --------: |
| qwen2.5-0.5b |  0.4921 |  0.3214 |  0.3103 |    0.3746 | 20.14 |     365.3 |
| qwen2.5-1.5b |  0.4966 |  0.3186 |  0.3801 |    0.3985 | 23.12 |     164.4 |
| qwen2.5-3b   |  0.4643 |  0.2633 |   0.246 |    0.3245 | 25.33 |     485.4 |
| qwen3-4b     |  0.4261 |  0.2307 |  0.2778 |    0.3115 | 27.85 |     276.6 |
| qwen2.5-7b   |  0.4915 |  0.2919 |  0.3625 |     0.382 | 28.94 |     161.7 |


## 输出与结果
- 默认在项目根目录生成/更新结果文件（位于 `./result/`）。
- 结果格式（CSV/JSON/纯文本）以脚本实现为准；若需导出其它格式，可在脚本中添加保存逻辑或另写解析脚本。


