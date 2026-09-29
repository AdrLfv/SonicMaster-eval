# HD Music SonicMaster preview

Submit `run_hd_music_preview_sm.sh` from kuma to restore the six clips selected
in `/work/vita/alefevre/programs/ARIEL/tmp/hd_music_sa3_attn_mlp_rk128_preview`.
It uses the local `checkpoints/model.safetensors` SonicMaster checkpoint without
a text prompt and writes the results below that preview directory.

Each of the 19 degradation folders in `sonicmaster/` contains six WAVs for each
of `degraded/`, `clean/`, `clean_decoded/`, and `restored/`. The clean-decoded
audio is the Stable Audio Open VAE encode/decode baseline used by SonicMaster.
