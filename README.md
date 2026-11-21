# Harness Tuning & Optimization Tool

Harness Tuning & Optimization Framework cho AI Coding Agent dựa trên nghiên cứu thực nghiệm Self-Harness, Harness Evolution, Meta-Harness, Darwin Gödel Machine và Evo-Harness.

Hệ thống cho phép vận hành một Coding Agent cụ thể (như **Pi Coding Agent**) với cấu hình Harness chỉ định (model, prompt, tools, runtime policy, skills) trên các bộ benchmark tiêu chuẩn (**Terminal-Bench-2.0** / **SWE-bench**), gom cụm lỗi từ trace thực thi, dùng **Dedicated Proposer Model** sinh bản vá có kiểm soát (bounded declarative patches), kiểm định qua **Non-Regression Gate** (Held-in + Held-out) và xuất kết quả ra **thư mục riêng biệt**, tuyệt đối không ghi đè lên harness gốc.

---

## 1. Prerequisites & Pinned Runtimes

- **Python**: `>= 3.10`
- **PyYAML**: `>= 6.0`
- **Pi Coding Agent**: `@earendil-works/pi-coding-agent` (tuỳ chọn khi chạy real environment)

---

## 2. Architecture & Layout

```
.
├── harness_tune/                  # Core optimization package
│   ├── core/
│   │   ├── manifest.py            # HarnessManifest & typed surfaces validation
│   │   ├── trace.py               # Trace recording & step data structures
│   │   ├── failure_miner.py       # Failure pattern clustering & symptoms
│   │   ├── proposer.py            # Dedicated Proposer model integration
│   │   ├── patcher.py             # Isolated declarative patcher
│   │   ├── regression_gate.py     # Statistical non-regression evaluation
│   │   └── registry.py            # Version lineage & Pareto archive
│   ├── agents/
│   │   ├── base.py                # AbstractCodingAgent
│   │   └── pi_agent.py            # Pi Coding Agent adapter (minimal 4-tool harness)
│   ├── benchmarks/
│   │   ├── base.py                # AbstractBenchmark interface
│   │   ├── terminal_bench.py      # Terminal-Bench-2.0 adapter (held-in/held-out)
│   │   └── swe_bench.py           # SWE-bench / SWE-bench Lite adapter
│   ├── engine.py                  # End-to-end HarnessEvolutionEngine
│   └── cli.py                     # CLI entrypoint
├── templates/
│   └── pi_agent_harness/          # Template harness tinh gọn chuẩn cho Pi Coding Agent
│       ├── configs/
│       │   ├── harness.manifest.json
│       │   ├── runtime.yaml
│       │   ├── tools.yaml
│       │   └── verification.yaml
│       ├── prompts/system.md
│       └── skills/self-evolve/SKILL.md
├── tests/                         # Unit tests (100% pass)
└── docs/                          # Tài liệu nghiên cứu và đặc tả
```

---

## 3. CLI Usage: Tối ưu hóa Harness cho Pi Coding Agent

### Chạy chu trình tối ưu hóa (Optimize Cycle)
```bash
python3 -m harness_tune.cli optimize \
  --manifest templates/pi_agent_harness/configs/harness.manifest.json \
  --agent pi \
  --target-model claude-3-5-sonnet \
  --proposer-model claude-3-7-sonnet \
  --benchmark terminal_bench \
  --output-dir ./optimized_pi_harnesses
```

**Các tham số:**
- `--manifest`: Đường dẫn file `harness.manifest.json` của harness gốc cần tối ưu.
- `--agent`: Agent đích (`pi`).
- `--target-model`: Model mục tiêu mà Worker Agent sử dụng (ví dụ: `qwen-2.5-coder-32b`, `claude-3-5-sonnet`, `gemini-3.8-flash`).
- `--proposer-model`: Model Proposer chuyên trách cao cấp (mặc định: `claude-3-7-sonnet`).
- `--benchmark`: Bộ benchmark đánh giá (`terminal_bench` hoặc `swe_bench`).
- `--output-dir`: Thư mục riêng biệt lưu trữ harness được tối ưu (không đè lên harness gốc).

### Chạy đánh giá độc lập (Benchmark only)
```bash
python3 -m harness_tune.cli benchmark \
  --manifest templates/pi_agent_harness/configs/harness.manifest.json \
  --agent pi \
  --benchmark terminal_bench
```

---

## 4. Development & Verification Commands

- **Chạy toàn bộ unit tests**:
  ```bash
  python3 -m unittest discover -s tests -p "*_test.py"
  ```
- **Kiểm tra cú pháp Python (Format check)**:
  ```bash
  python3 -m py_compile harness_tune/*.py harness_tune/**/*.py tests/*.py
  ```
