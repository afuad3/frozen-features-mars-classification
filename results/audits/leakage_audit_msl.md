# MSL leakage audit (§6.2)

**RESULT: PASS ✅**

## Dataset-artifact leakage policy (user decision: drop train duplicates)
- Pre-dedup: 2 TRAIN image(s) pixel-identical to a val/test image (recurring 'Artifact' frame in the official v2.1 split).
- Policy `dedup_train_against_eval=True`: dropped those TRAIN copies; val/test kept intact. Embedded train size: 1998 (was 2000).
- Dropped train image_ids: ['0041ML0001800030101349I01_DRCL', '0165MH0001880200102035P01_DRCL']

## Checks (on the POST-dedup embedded set)
- (i) each image in exactly one split: OK
- (i-b) no base product across >1 split: OK
- (ii) sol ranges per split: {'test': {'min': 1922, 'max': 2224}, 'train': {'min': 3, 'max': 948}, 'val': {'min': 952, 'max': 1918}}
      chronological (test sols >= train max): YES
- (iii) no exact file-hash across splits: OK
- (iii) no exact pixel-hash across splits: OK
