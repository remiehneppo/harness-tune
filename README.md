# Harness Tuning & Optimization Tool

Harness Tuning & Optimization Framework cho AI Coding Agent dựa trên nghiên cứu thực nghiệm Self-Harness, Harness Evolution, Meta-Harness, Darwin Gödel Machine và Evo-Harness.

Hệ thống cho phép vận hành **Pi Coding Agent** (`pi`) trên thiết bị với model local (như **`arex-2`** qua `http://127.0.0.1:8080/v1`), chạy đánh giá trên cả bộ **Terminal-Bench 2.0 rút gọn (6 task)** lẫn **Full Official Terminal-Bench 2.0 (89 task Docker)** từ `laude-institute/terminal-bench-2`, gom cụm lỗi từ NDJSON execution traces, dùng **Gemini 3.8 Flash** (`gemini-3.8-flash-high`) qua **`agy` CLI** làm Orchestrator/Proposer sinh bản vá có kiểm soát (bounded declarative patches), kiểm định qua **Non-Regression Gate** ($\Delta p_{ho} \ge 0$) và xuất kết quả ra thư mục riêng biệt mà không ghi đè harness gốc.

---

## 1. Prerequisites & Pinned Runtimes

- **Python**: `>= 3.10`
- **PyYAML**: `>= 5.4`
- **Docker Engine**: cần khi chạy `terminal_bench_2_full` (89 task container chính thức)
- **Pi Coding Agent**: `pi` CLI (`~/.local/bin/pi` & `~/.pi/agent/install/releases/<version>`) cấu hình provider `local` / model `arex-2`
- **Antigravity CLI (`agy`)**: `~/.local/bin/agy` dùng cho orchestrator model `gemini-3.8-flash-high`

---

## 2. Architecture & Layout

```
.
├── harness_tune/                  # Core optimization package
│   ├── core/
│   │   ├── agy_client.py          # Wrapper gọi `agy` CLI (Gemini 3.8 Flash orchestrator)
│   │   ├── manifest.py            # HarnessManifest & typed surfaces validation
│   │   ├── trace.py               # Trace recording & step data structures
│   │   ├── failure_miner.py       # Failure pattern clustering & symptoms
│   │   ├── proposer.py            # Dedicated Proposer model integration
│   │   ├── patcher.py             # Isolated declarative patcher
│   │   ├── regression_gate.py     # Statistical non-regression evaluation
│   │   └── registry.py            # Version lineage & Pareto archive
│   ├── agents/
│   │   ├── base.py                # AbstractCodingAgent
│   │   └── pi_agent.py            # Pi Coding Agent adapter (host & Docker container NDJSON)
│   ├── benchmarks/
│   │   ├── base.py                # AbstractBenchmark interface
│   │   ├── terminal_bench.py      # Terminal-Bench-2.0 light suite (6 seeded tasks)
│   │   ├── terminal_bench2_docker.py # Official Terminal-Bench 2.0 Docker adapter (89 tasks)
│   │   └── swe_bench.py           # SWE-bench / SWE-bench Lite adapter
│   ├── engine.py                  # End-to-end HarnessEvolutionEngine (multi-round + resume)
│   └── cli.py                     # CLI entrypoint
├── templates/
│   └── pi_agent_harness/          # Template harness tinh gọn chuẩn cho Pi Coding Agent
└── tests/                         # Unit tests
```

---

## 3. CLI Usage: Tinh chỉnh Pi Agent + `arex-2` bằng `agy` (`gemini-3.8-flash-high`)

### Bước 1: Chạy Baseline với Full User Profile trên 20 task đầu Terminal-Bench 2.0 (20 phút/task)
```bash
python3 -m harness_tune.cli benchmark \
  --manifest templates/pi_user_harness/configs/harness_arex2_user.manifest.json \
  --agent pi \
  --agent-mode real \
  --user-profile \
  --target-model arex-2 \
  --provider local \
  --benchmark terminal_bench_2_full \
  --dataset-path ./datasets/terminal-bench-2 \
  --limit 20 \
  --timeout 1200 \
  --output-dir ./tb2_user_20_baseline
```

### Bước 1b: Hoặc chạy Baseline Full 89 task Docker (có checkpoint/resume)
```bash
python3 -m harness_tune.cli benchmark \
  --manifest templates/pi_agent_harness/configs/harness_arex2.manifest.json \
  --agent pi \
  --agent-mode real \
  --target-model arex-2 \
  --provider local \
  --benchmark terminal_bench_2_full \
  --dataset-path ./datasets/terminal-bench-2 \
  --timeout 300 \
  --output-dir ./tb2_full_baseline
```

### Bước 2: Tối ưu từ các trace baseline đã lưu (`--baseline-traces`) + Fast Screen
```bash
python3 -m harness_tune.cli optimize \
  --manifest templates/pi_agent_harness/configs/harness_arex2.manifest.json \
  --agent pi \
  --agent-mode real \
  --target-model arex-2 \
  --provider local \
  --orchestrator agy \
  --proposer-model gemini-3.8-flash-high \
  --benchmark terminal_bench_2_full \
  --dataset-path ./datasets/terminal-bench-2 \
  --baseline-traces ./tb2_full_baseline/baseline_traces.json \
  --fast-screen \
  --rounds 1 \
  --max-proposals 2 \
  --timeout 300 \
  --output-dir ./tb2_full_optimized
```

### Chạy nhanh bộ nhẹ (`terminal_bench`, 6 task)
```bash
python3 -m harness_tune.cli optimize \
  --manifest templates/pi_agent_harness/configs/harness_arex2.manifest.json \
  --agent pi \
  --agent-mode real \
  --target-model arex-2 \
  --provider local \
  --orchestrator agy \
  --proposer-model gemini-3.8-flash-high \
  --benchmark terminal_bench \
  --rounds 1 \
  --max-proposals 2 \
  --timeout 180 \
  --output-dir ./optimized_arex2_v2
```

---

## 4. Development & Verification Commands

- **Chạy toàn bộ unit tests**:
  ```bash
  python3 -m unittest discover -s tests -p "*_test.py"
  ```
- **Kiểm tra cú pháp Python**:
  ```bash
  shopt -s globstar && python3 -m py_compile harness_tune/**/*.py tests/*.py
  ```
