# Synthetic XRF data

`mosaic_Fe.tiff` is a 1040 x 1040 float32 Fe map sampled at 0.25 um/pixel,
representing a 260 x 260 um coarse mosaic. The three `fine_spot_*_Fe.tiff`
files are 200 x 200 float32 maps sampled from the same synthetic scene at
0.05 um/pixel. Each therefore covers a 10 x 10 um selected region at five
times the mosaic's linear resolution.

`metadata.json` records each fine scan's physical center in the mosaic frame.
Regenerate the files with:

```bash
pixi run python scripts/generate_fake_data.py
```
