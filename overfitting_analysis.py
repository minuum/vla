import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_all_versions():
    base_dir = "RoboVLMs_upstream/runs/v2_classification/kosmos/mobile_vla_v2_classification/2026-02-17/v2-classification-9cls/v2-classification-9cls/"
    versions = sorted(glob.glob(os.path.join(base_dir, "version_*")))
    
    results = []
    
    for v in versions:
        event_files = glob.glob(os.path.join(v, "events.out.tfevents.*"))
        if not event_files: continue
        
        # Load the latest event file in this version
        latest_file = max(event_files, key=os.path.getmtime)
        ea = EventAccumulator(latest_file)
        ea.Reload()
        tags = ea.Tags()['scalars']
        
        if 'val_loss' in tags and 'train_loss' in tags:
            val_events = ea.Scalars('val_loss')
            train_events = ea.Scalars('train_loss')
            
            # Map val_loss to steps
            for v_ev in val_events:
                # Find nearest train_loss before this val step
                train_val = 0
                for t_ev in train_events:
                    if t_ev.step <= v_ev.step:
                        train_val = t_ev.value
                    else:
                        break
                
                results.append({
                    'version': os.path.basename(v),
                    'step': v_ev.step,
                    'val_loss': v_ev.value,
                    'train_loss': train_val
                })

    # Sort and print
    print("\n" + "="*70)
    print(f"{'Version':<12} | {'Step':<8} | {'Train Loss':<12} | {'Val Loss':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['version']:<12} | {r['step']:<8} | {r['train_loss']:>12.6f} | {r['val_loss']:>12.6f}")
    print("="*70)

if __name__ == "__main__":
    analyze_all_versions()
