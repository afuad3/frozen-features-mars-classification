# MSL data audit - Mars surface image (Curiosity rover) labeled data set version 2.1

- Zenodo record: 4033453
- Total released images (incl. augmented train): **6820** {'train': 5920, 'test': 600, 'val': 300}
- Originals (pre-dedup, paper's benchmark): **2900** {'train': 2000, 'test': 600, 'val': 300}
  - expected {'train': 2000, 'val': 300, 'test': 600, 'total': 2900} -> **MATCH**
- **Embedded set (post train-dedup): 2898** {'train': 1998, 'test': 600, 'val': 300} (dropped 2 train duplicate(s) of val/test per leakage policy)
- Classes present: **19** (declared 19)
- Filenames without parseable sol/instrument: **0**

## Image properties (sampled, originals)
- Sizes: {(227, 227): 50}
- Modes/channels: {'RGB': 50}

## Sol range per split (chronological check, originals)
| split   |   min |   max |   count |
|:--------|------:|------:|--------:|
| test    |  1922 |  2224 |     600 |
| train   |     3 |   948 |    1998 |
| val     |   952 |  1918 |     300 |

## Instrument distribution per split (originals)
| instrument    |   test |   train |   val |
|:--------------|-------:|--------:|------:|
| mahli         |    163 |     479 |    85 |
| mastcam_left  |    222 |     841 |   108 |
| mastcam_right |    215 |     678 |   107 |

## Class distribution (originals)
- Nearby surface: 1008 (34.8%)
- Artifact: 406 (14.0%)
- Close-up rock: 373 (12.9%)
- Distant landscape: 197 (6.8%)
- Sand: 123 (4.2%)
- Sun: 115 (4.0%)
- Layers: 105 (3.6%)
- Mastcam cal target: 100 (3.5%)
- Other rover part: 86 (3.0%)
- Float: 80 (2.8%)
- Drill hole: 65 (2.2%)
- Wheel: 56 (1.9%)
- DRT spot: 47 (1.6%)
- Light-toned veins: 42 (1.4%)
- Wheel joint: 33 (1.1%)
- Night sky: 23 (0.8%)
- Wheel tracks: 15 (0.5%)
- DRT: 14 (0.5%)
- Arm cover: 10 (0.3%)

## Sample filenames
```
0042MR0001830110101823Q01_DRCL.jpg
0442MH0001520020200158I01_DRCL.jpg
0948MH0004910060303925I01_DRCL.jpg
0037ML0000990110101254Q01_DRCL.jpg
0042MR0001810030101417Q01_DRCL.jpg
0042MR0001810030101428Q01_DRCL.jpg
0289ML0009620000106354M00_DRCL.jpg
0393MR0016220130302162E01_DRCL.jpg
```