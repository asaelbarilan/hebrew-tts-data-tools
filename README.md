# tts-data-tools

Reusable Hebrew TTS normalizer + dataset preparation utilities copied from `clean_f5`.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/smoke_test_normalizer.py
```

## UI

```bash
set PYTHONPATH=.
python ui/audit_app.py
```

Optionally set data path:

```bash
set DATA_PATH=C:\path\to\your\dataset
```
