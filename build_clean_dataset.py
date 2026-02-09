import json
import os
import logging
import h5py
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TARGET_SR = 44100
SHARD_SIZE = 1000

class DatasetBuilder:
    def __init__(self, base_dir, clean_dir):
        self.base_dir = Path(base_dir)
        self.clean_dir = Path(clean_dir)
        self.train_dir = self.base_dir / 'train'
        self.test_dir = self.base_dir / 'test'
        
        self.train_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)
        
        self.train_metadata = []
        self.test_metadata = []
        
        self.train_audio_index = 0
        self.test_audio_index = 0
        
        self.train_shard_idx = 0
        self.test_shard_idx = 0
        
        self.train_shard_data = []
        self.test_shard_data = []
        
        self.processed_train_sources = set()
        self.processed_test_sources = set()
    
    def load_audio_and_resample(self, audio_path):
        """Load audio file and resample to target sample rate if necessary."""
        try:
            audio, original_sr = sf.read(audio_path)
            
            if original_sr != TARGET_SR:
                logger.info(f"Resampling {audio_path.name} from {original_sr}Hz to {TARGET_SR}Hz")
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=TARGET_SR)
            
            if audio.ndim == 1:
                audio = audio[:, np.newaxis]
            elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
                audio = audio.T
            
            return audio, original_sr
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None, None
    
    def save_shard(self, split, shard_idx, shard_data):
        """Save a shard of audio data to HDF5."""
        if not shard_data:
            return
        
        output_dir = self.train_dir if split == 'train' else self.test_dir
        shard_path = output_dir / f"shard_{shard_idx:04d}.h5"
        
        logger.info(f"Saving {split} shard {shard_idx} with {len(shard_data)} samples to {shard_path}")
        
        shard_path_str = str(shard_path)
        with h5py.File(shard_path, 'w') as f:
            for idx, (audio_data, metadata) in enumerate(shard_data):
                source_id = metadata.get('source_id', f"sample_{idx:05d}")
                dataset_key = f"sample_{idx:05d}_{source_id}"
                grp = f.create_group(dataset_key)
                grp.create_dataset('audio', data=audio_data, compression='gzip', compression_opts=4)
                
                clean_audio_path = f"{shard_path_str}::/{dataset_key}"
                metadata['clean_audio_path'] = clean_audio_path
                metadata['clean_audio_dataset'] = dataset_key
                metadata['clean_audio_shard'] = shard_path_str
                
                for key, value in metadata.items():
                    if value is None:
                        continue
                    if isinstance(value, (list, dict)):
                        grp.attrs[key] = json.dumps(value)
                    else:
                        grp.attrs[key] = value
    
    def process_item(self, item):
        """Process a single item from the dataset."""
        split = item.get('split')
        if split not in ['train']: # CHANGE THIS TO 'test' OR ADD IT TO THE LIST WHEN YOU WANT TO BUILD THE TEST SET
            return
        
        source_id = item.get('source_id')
        if not source_id:
            return
        
        if split == 'train':
            if source_id in self.processed_train_sources:
                return
            self.processed_train_sources.add(source_id)
            current_index = self.train_audio_index
            self.train_audio_index += 1
        else:
            if source_id in self.processed_test_sources:
                return
            self.processed_test_sources.add(source_id)
            current_index = self.test_audio_index
            self.test_audio_index += 1
        
        source_file = self.clean_dir / f"{source_id}.flac"
        if not source_file.exists():
            logger.warning(f"Audio file not found: {source_file}")
            return
        
        audio_data, original_sr = self.load_audio_and_resample(source_file)
        if audio_data is None:
            return
        
        original_length = int(item.get('duration', 0) * original_sr)
        
        metadata_entry = {
            'name': item.get('name'),
            'duration': item.get('duration'),
            'genres': item.get('genres'),
            'vocalinstrumental': item.get('vocalinstrumental'),
            'gender': item.get('gender'),
            'vartags': item.get('vartags'),
            'scores': item.get('scores'),
            'genre_group': item.get('genre_group'),
            'clip_start': item.get('clip_start'),
            'clip_end': item.get('clip_end'),
            'source_id': source_id,
            'clean_audio_path': None,
            'original_length': original_length,
            'sample_rate': TARGET_SR,
            'original_sample_rate': original_sr,
            'clean_audio_index': current_index
        }
        
        if split == 'train':
            self.train_metadata.append(metadata_entry)
            self.train_shard_data.append((audio_data, metadata_entry))
            
            if len(self.train_shard_data) >= SHARD_SIZE:
                self.save_shard('train', self.train_shard_idx, self.train_shard_data)
                self.train_shard_idx += 1
                self.train_shard_data = []
        else:
            self.test_metadata.append(metadata_entry)
            self.test_shard_data.append((audio_data, metadata_entry))
            
            if len(self.test_shard_data) >= SHARD_SIZE:
                self.save_shard('test', self.test_shard_idx, self.test_shard_data)
                self.test_shard_idx += 1
                self.test_shard_data = []
        
        logger.info(f"Processed {source_id} for {split} set (index: {current_index})")
    
    def save_metadata(self):
        """Save metadata to JSONL files."""
        if self.train_metadata:
            train_jsonl_path = self.train_dir / 'metadata.jsonl'
            with open(train_jsonl_path, 'w') as f:
                for entry in self.train_metadata:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"Saved train metadata to {train_jsonl_path} ({len(self.train_metadata)} entries)")
        
        if self.test_metadata:
            test_jsonl_path = self.test_dir / 'metadata.jsonl'
            with open(test_jsonl_path, 'w') as f:
                for entry in self.test_metadata:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"Saved test metadata to {test_jsonl_path} ({len(self.test_metadata)} entries)")
    
    def finalize(self):
        """Save remaining shards and metadata."""
        if self.train_shard_data:
            self.save_shard('train', self.train_shard_idx, self.train_shard_data)
        
        if self.test_shard_data:
            self.save_shard('test', self.test_shard_idx, self.test_shard_data)
        
        self.save_metadata()
        
        logger.info("Dataset building completed:")
        logger.info(f"  Train: {len(self.train_metadata)} samples in {self.train_shard_idx + (1 if self.train_shard_data else 0)} shards")
        logger.info(f"  Test: {len(self.test_metadata)} samples in {self.test_shard_idx + (1 if self.test_shard_data else 0)} shards")


def main():
    base_dir = '/mnt/ssd2/datasets/SonicMasterDataset'
    clean_dir = os.path.join(base_dir, 'clean')
    jsonl_path = os.path.join(base_dir, 'SonicMasterdataset.jsonl')
    
    builder = DatasetBuilder(base_dir, clean_dir)
    
    logger.info("Starting dataset building process...")
    logger.info("Reading from: %s", jsonl_path)
    
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, desc="Processing dataset"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                builder.process_item(item)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON line: {e}")
    
    builder.finalize()


if __name__ == '__main__':
    main()
