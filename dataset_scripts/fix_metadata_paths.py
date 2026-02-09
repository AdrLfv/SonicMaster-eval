import json
import sys

def fix_paths(input_jsonl, output_jsonl, old_path, new_path):
    """Replace old paths with new paths in JSONL metadata file."""
    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
        
        count = 0
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            entry = json.loads(line)
            
            # Update paths
            if 'clean_audio_path' in entry and entry['clean_audio_path']:
                entry['clean_audio_path'] = entry['clean_audio_path'].replace(old_path, new_path)
            
            if 'clean_audio_shard' in entry and entry['clean_audio_shard']:
                entry['clean_audio_shard'] = entry['clean_audio_shard'].replace(old_path, new_path)
            
            outfile.write(json.dumps(entry) + '\n')
            count += 1
        
        print(f"Processed {count} entries")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python fix_metadata_paths.py <input_jsonl> <output_jsonl> <old_path> <new_path>")
        sys.exit(1)
    
    input_jsonl = sys.argv[1]
    output_jsonl = sys.argv[2]
    old_path = sys.argv[3]
    new_path = sys.argv[4]
    
    fix_paths(input_jsonl, output_jsonl, old_path, new_path)
