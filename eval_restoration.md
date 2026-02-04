# Encoding Commands

```bash
sbatch run_encode.sh airy
sbatch run_encode.sh big
sbatch run_encode.sh boom
sbatch run_encode.sh bright
sbatch run_encode.sh clarity
sbatch run_encode.sh clip
sbatch run_encode.sh dark
sbatch run_encode.sh mic
sbatch run_encode.sh comp
sbatch run_encode.sh mix
sbatch run_encode.sh mud
sbatch run_encode.sh punch
sbatch run_encode.sh real
sbatch run_encode.sh small
sbatch run_encode.sh stereo
sbatch run_encode.sh vocal
sbatch run_encode.sh volume
sbatch run_encode.sh warm
sbatch run_encode.sh xband
```

# Restoration Commands (No Prompt)

```bash
sbatch run_restoration.sh airy
sbatch run_restoration.sh big
sbatch run_restoration.sh boom
sbatch run_restoration.sh bright
sbatch run_restoration.sh clarity
sbatch run_restoration.sh clip
sbatch run_restoration.sh comp
sbatch run_restoration.sh dark
sbatch run_restoration.sh mic
sbatch run_restoration.sh mix
sbatch run_restoration.sh mud
sbatch run_restoration.sh punch
sbatch run_restoration.sh real
sbatch run_restoration.sh small
sbatch run_restoration.sh stereo
sbatch run_restoration.sh vocal
sbatch run_restoration.sh volume
sbatch run_restoration.sh warm
sbatch run_restoration.sh xband
```

# Restoration Commands (With Prompt)

```bash
sbatch run_restoration.sh airy "Restore the airy high-end texture and remove the dull veil."
sbatch run_restoration.sh big "Can you remove the excess reverb in this audio, please?"
sbatch run_restoration.sh boom "Tighten the low end and remove the excessive boominess."
sbatch run_restoration.sh bright "Bring back the top-end sparkle without making it harsh."
sbatch run_restoration.sh clarity "Clear up the smeared transients and improve articulation."
sbatch run_restoration.sh clip "Reduce the clipping and reconstruct the lost audio, please."
sbatch run_restoration.sh comp "Undo the heavy compression and recover the natural dynamics."
sbatch run_restoration.sh dark "Make the tone fuller and less sharp."
sbatch run_restoration.sh mic "Remove the microphone coloration and make the tone neutral again."
sbatch run_restoration.sh mix "Reduce the artificial reverb blend and keep only natural ambience."
sbatch run_restoration.sh mud "Remove the muddiness and open up the midrange."
sbatch run_restoration.sh punch "Reduce the punch and reconstruct the lost audio, please."
sbatch run_restoration.sh real "Please, reduce the strong echo in this song."
sbatch run_restoration.sh small "Clean this off any echoes!"
sbatch run_restoration.sh stereo "Restore the stereo width and fix the imaging."
sbatch run_restoration.sh vocal "Bring the vocals back to their natural level and tone."
sbatch run_restoration.sh volume "Restore the proper loudness level."
sbatch run_restoration.sh warm "Make the sound warmer and more inviting."
sbatch run_restoration.sh xband "Restore the full-spectrum balance and undo the multiband EQ color."
```

# Evaluation Commands for Restored Audio (No Prompt)

```bash
sbatch run_evaluate_aae.sh airy restored
sbatch run_evaluate_aae.sh big restored
sbatch run_evaluate_aae.sh boom restored
sbatch run_evaluate_aae.sh bright restored
sbatch run_evaluate_aae.sh clarity restored
sbatch run_evaluate_aae.sh clip restored
sbatch run_evaluate_aae.sh comp restored
sbatch run_evaluate_aae.sh dark restored
sbatch run_evaluate_aae.sh mic restored
sbatch run_evaluate_aae.sh mix restored
sbatch run_evaluate_aae.sh mud restored
sbatch run_evaluate_aae.sh punch restored
sbatch run_evaluate_aae.sh real restored
sbatch run_evaluate_aae.sh small restored
sbatch run_evaluate_aae.sh stereo restored
sbatch run_evaluate_aae.sh vocal restored
sbatch run_evaluate_aae.sh volume restored
sbatch run_evaluate_aae.sh warm restored
sbatch run_evaluate_aae.sh xband restored
```

# Evaluation Commands for Restored Audio (With Prompt)

```bash
sbatch run_evaluate_aae.sh airy restored_prompt
sbatch run_evaluate_aae.sh big restored_prompt
sbatch run_evaluate_aae.sh boom restored_prompt
sbatch run_evaluate_aae.sh bright restored_prompt
sbatch run_evaluate_aae.sh clarity restored_prompt
sbatch run_evaluate_aae.sh clip restored_prompt
sbatch run_evaluate_aae.sh comp restored_prompt
sbatch run_evaluate_aae.sh dark restored_prompt
sbatch run_evaluate_aae.sh mic restored_prompt
sbatch run_evaluate_aae.sh mix restored_prompt
sbatch run_evaluate_aae.sh mud restored_prompt
sbatch run_evaluate_aae.sh punch restored_prompt
sbatch run_evaluate_aae.sh real restored_prompt
sbatch run_evaluate_aae.sh small restored_prompt
sbatch run_evaluate_aae.sh stereo restored_prompt
sbatch run_evaluate_aae.sh vocal restored_prompt
sbatch run_evaluate_aae.sh volume restored_prompt
sbatch run_evaluate_aae.sh warm restored_prompt
sbatch run_evaluate_aae.sh xband restored_prompt
```
