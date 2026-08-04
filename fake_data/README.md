# Synthetic XRF data

`mosaic_Fe.tiff`, `mosaic_Mn.tiff`, and `mosaic_Cr.tiff` are aligned 1040 x
1040 float32 element maps sampled at 0.25 um/pixel, each representing a 260 x
260 um coarse mosaic. Each selected region has matching Fe, Mn, and Cr
`fine_spot_*` TIFFs. They are 200 x 200 float32 maps sampled from the same
synthetic scene at 0.05 um/pixel, covering 10 x 10 um at five times the
mosaic's linear resolution.

`metadata.json` records each fine scan's physical center in the mosaic frame.
Regenerate the files with:

```bash
pixi run python scripts/generate_fake_data.py
```
