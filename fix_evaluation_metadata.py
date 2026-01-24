import json
import shutil

input_file = "/work/vita/datasets/audio/sonicmaster/audios/restored_with_SM_model_specific_punch_prompt/inference_20260124_134710/evaluation_metadata.jsonl"
backup_file = input_file + ".backup"
temp_file = input_file + ".tmp"

shutil.copy(input_file, backup_file)
print(f"Backup created: {backup_file}")

with open(input_file, 'r') as inf, open(temp_file, 'w') as outf:
    for line in inf:
        entry = json.loads(line)
        deg_spec = entry.get("degradation_spec", "")
        deg_group = entry.get("degradation_group", "")
        entry["degradation_name"] = f"{deg_group}_sonicmaster_{deg_spec}"
        outf.write(json.dumps(entry) + '\n')

shutil.move(temp_file, input_file)
print(f"Fixed metadata saved to: {input_file}")
