# Local synthetic data

This directory intentionally contains no committed data files. It is reserved
for locally generated synthetic mosaics, per-element fine-scan maps, merged
fine scans, and analysis outputs. All contents except this README are ignored
by Git.

Generate the local coarse and per-element fine data with:

```bash
pixi run python scripts/generate_fake_data.py
```

Then create merged RGB fine scans with:

```bash
pixi run python scripts/merge_fine_scans.py
```
