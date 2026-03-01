import os
import glob
from datetime import datetime

def format_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

def find_checkpoints(base_dir):
    ckpt_files = []
    for root, dirs, files in os.walk(base_dir):
        # Omit some large hidden directories if any
        if '.git' in dirs:
            dirs.remove('.git')
        for file in files:
            if file.endswith('.ckpt'):
                full_path = os.path.join(root, file)
                try:
                    stats = os.stat(full_path)
                    ckpt_files.append({
                        'path': full_path,
                        'name': file,
                        'size': stats.st_size,
                        'mtime': stats.st_mtime,
                        'dir': root
                    })
                except OSError:
                    pass
    return ckpt_files

if __name__ == '__main__':
    base_dir = '/home/billy/25-1kp/vla'
    print(f"Scanning for .ckpt files in {base_dir}...\n")
    checkpoints = find_checkpoints(base_dir)
    
    # Sort by directory, then by modification time
    checkpoints.sort(key=lambda x: (x['dir'], x['mtime']))
    
    total_size = sum(c['size'] for c in checkpoints)
    print(f"Total .ckpt files found: {len(checkpoints)}")
    print(f"Total size of all checkpoints: {format_size(total_size)}\n")
    
    # Group by directory and print
    grouped = {}
    for c in checkpoints:
        if c['dir'] not in grouped:
            grouped[c['dir']] = []
        grouped[c['dir']].append(c)
        
    for d, files in grouped.items():
        dir_size = sum(f['size'] for f in files)
        print(f"Directory: {d}")
        print(f"  Total size in this dir: {format_size(dir_size)}")
        print(f"  Files:")
        for f in files:
            date_str = datetime.fromtimestamp(f['mtime']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"    - {f['name']:<40} | Size: {format_size(f['size']):<10} | Modified: {date_str}")
        print("-" * 80)
