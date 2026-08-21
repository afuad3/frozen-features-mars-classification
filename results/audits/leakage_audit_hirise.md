# HiRISE leakage audit (§6.1)

**RESULT: PASS ✅**

## Checks
- (a) split counts 6997/2025/1793: OK ({'train': 6997, 'val': 2025, 'test': 1793})
- (b) no source-image shared across splits: OK
- (c) no landmark family shared across splits: OK
- (d) no exact file-hash across splits: OK
- (d) no exact pixel-hash across splits: OK

## Notes
- Test set is the released unaugmented set; train/val use one original per landmark.
- Near-duplicate (perceptual) detection beyond exact pixel hashing is a documented limitation.
