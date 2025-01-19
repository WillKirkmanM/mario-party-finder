import os
from pathlib import Path
import hashlib
from collections import defaultdict

def get_file_hash(filepath):
    """Calculate SHA-256 hash of file to identify true duplicates"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def remove_duplicates(data_dir):
    data_path = Path(data_dir)
    duplicates_found = 0
    duplicates_removed = 0
    
    files_by_name = defaultdict(list)
    
    for game_dir in data_path.glob('*'):
        if game_dir.is_dir():
            print(f"Scanning {game_dir.name}...")
            for file_path in game_dir.glob('*.*'):
                if file_path.suffix.lower() in ['.webp', '.jpg', '.jpeg', '.png']:
                    base_name = file_path.stem.split('_')[0]
                    files_by_name[base_name].append(file_path)
    
    for base_name, file_paths in files_by_name.items():
        if len(file_paths) > 1:
            no_suffix_files = [f for f in file_paths if '_' not in f.stem]
            
            if len(no_suffix_files) > 1:
                duplicates_found += len(no_suffix_files) - 1
                file_hashes = {}
                for file_path in no_suffix_files:
                    file_hash = get_file_hash(file_path)
                    if file_hash in file_hashes:
                        print(f"Removing duplicate: {file_path}")
                        os.remove(file_path)
                        duplicates_removed += 1
                    else:
                        file_hashes[file_hash] = file_path
    
    print(f"\nSummary:")
    print(f"Duplicates found: {duplicates_found}")
    print(f"Duplicates removed: {duplicates_removed}")

if __name__ == "__main__":
    data_dirs = ['data/train', 'data/val', 'data/test']
    for dir_path in data_dirs:
        print(f"\nProcessing {dir_path}...")
        remove_duplicates(dir_path)