# Day 11 — Controlled Agent Security (2026)

**Họ tên:** Nguyễn Tiến Đạt · **MSSV:** 2A202603752 · **Lớp:** K5-L3B

> 👤 Bài tập **cá nhân** (1 người / 1 MSSV).
> Hướng dẫn làm bài: [`CHECKPOINTS.md`](CHECKPOINTS.md) · Nộp bài: [`SUBMISSION.md`](SUBMISSION.md) · Tiêu chí chấm: [`RUBRIC.md`](RUBRIC.md) · Quy định: [`RULES.md`](RULES.md)

---

## Cách chạy ngắn

Yêu cầu: Python 3.10+ · `OPENROUTER_API_KEY` (Blue) và key Red (`OPENAI_API_KEY` hoặc `GOOGLE_API_KEY`).

```bash
python3 -m venv .venv
source .venv/bin/activate
cp .env.example .env          # điền key + RED_TEAM_PROVIDER
pip install -r requirements.txt

python src/main.py --part 2   # CP2 — guardrails
python src/main.py --part 3   # CP3 — outputs/results.json
python src/main.py --part 4   # CP4 — outputs/attack_results.json
python -m pytest tests/smoke tests/public -q
python scripts/grade.py --submission-dir . --out outputs/grade_report.json
```

Windows: `python -m venv .venv` · `.\.venv\Scripts\Activate.ps1` · `Copy-Item .env.example .env`

Lệnh ↔ checkpoint ↔ file sinh ra, gồm cả bonus B1/B2 (lệnh `--part 4` sinh đồng thời `unsafe_*` = Red/B1 và `guards_*` = Red Advance/B2): xem [Phụ lục A](CHECKPOINTS.md#phụ-lục-a--lệnh--checkpoint--file-sinh-ra).

## Lần chạy đã nộp

- Blue: OpenRouter `liquid/lfm-3-1.7b:free` (khóa cứng trong `src/core/config.py`)
- Red / Red Advance: `openai` / `gpt-4o-mini`
- `outputs/` chứa artifact từ lần chạy thật — chạy lại `--part 3|4` sẽ ghi đè.
