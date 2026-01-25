## Compute FAD Scores against clean set

### Compare Clean vs Restored Audio

CLI version:
```bash
fadtk clap-laion-music \
    /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
    /work/vita/datasets/audio/sonicmaster/audios/restored_dark/inference_20260124_170346
```

Robust version:
```bash
python evaluation/fadtk_robust.py \
  --baseline /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --eval /work/vita/datasets/audio/sonicmaster/audios/restored_dark/inference_20260124_170346 \
  --model clap-laion-music
```

### Compare Clean vs Degraded Audio (Baseline)

CLI version:
```bash
fadtk clap-laion-music \
    /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
    /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded
```

Robust version:
```bash
python evaluation/fadtk_robust.py \
  --baseline /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster \
  --eval /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded \
  --model clap-laion-music
```

## Compute FAD Scores against FMA Pop Baseline

### Compare FMA Pop vs Restored Audio

CLI version:
```bash
fadtk clap-laion-music \
    fma_pop \
    /work/vita/datasets/audio/sonicmaster/audios/restored_dark/inference_20260124_170346
```

Robust version:
```bash
python evaluation/fadtk_robust.py \
  --baseline fma_pop \
  --eval /work/vita/datasets/audio/sonicmaster/audios/restored_dark/inference_20260124_170346 \
  --model clap-laion-music
```

### Compare FMA Pop vs Degraded Audio (Baseline)

CLI version:
```bash
fadtk clap-laion-music \
    fma_pop \
    /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded
```

Robust version:
```bash
python evaluation/fadtk_robust.py \
  --baseline fma_pop \
  --eval /work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster_dark_degraded \
  --model clap-laion-music
```