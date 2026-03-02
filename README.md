# tts-data-tools

Reusable Hebrew TTS normalizer + dataset preparation utilities copied from `clean_f5`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Recommended workflow order

For TTS training, use this order by default:

1. **Prepare dataset first** (segment/filter/build metadata)
2. **Normalize text second** (on final text used for training)
3. Export final train/val artifacts

Why: if you normalize too early, later segmentation/filtering may still change text/audio alignment.

### When to normalize earlier

Normalize earlier only if your prep logic depends on normalized text (for example, text-based filtering or duration heuristics that require expanded numbers/abbreviations). In that case: normalize once before filtering, then re-check alignment.

## Files included

- `data_prep/prepare_ivritai.py` - primary dataset preparation script
- `normalizer/` - Hebrew normalization logic and resources (copied from `heb_norm`)
- `ui/audit_app.py` - UI to inspect raw vs normalized text with audio
- `scripts/smoke_test_normalizer.py` - quick sanity test

## Quick sanity test

```bash
python scripts/smoke_test_normalizer.py
```

## Data preparation

Run the prep script first and produce your dataset artifacts:

```bash
python data_prep/prepare_ivritai.py
```

After prep completes, run normalization on the text you will actually train with (either in your prep flow or as a post-step).

## UI audit

```bash
set PYTHONPATH=.
python ui/audit_app.py
```

Optionally set data path:

```bash
set DATA_PATH=C:\path\to\your\dataset
```
