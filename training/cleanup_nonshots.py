"""
Clean up non-shot labels from training data JSON files
Keeps only 'shot' and 'beep' labels
"""
import json
import os
from pathlib import Path

def cleanup_json_files(directory):
    """Remove all non_shot labels from JSON files in directory"""

    json_files = list(Path(directory).glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    print(f"Found {len(json_files)} JSON files\n")

    total_removed = 0

    for json_path in sorted(json_files):
        try:
            # Read JSON
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Count labels before
            original_count = len(data.get('labels', []))
            nonshot_count = sum(1 for l in data.get('labels', []) if l.get('type') == 'non_shot')

            if nonshot_count == 0:
                print(f"✓ {json_path.name}: No non-shots to remove")
                continue

            # Filter out non_shot labels
            data['labels'] = [l for l in data.get('labels', []) if l.get('type') != 'non_shot']

            # Update counts
            shot_count = sum(1 for l in data['labels'] if l.get('type') == 'shot')
            beep_count = sum(1 for l in data['labels'] if l.get('type') == 'beep')

            data['total_shots'] = shot_count
            data['total_labels'] = len(data['labels'])

            # Write back
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

            total_removed += nonshot_count
            print(f"✓ {json_path.name}: Removed {nonshot_count} non-shots, kept {shot_count} shots + {beep_count} beeps")

        except Exception as e:
            print(f"✗ {json_path.name}: Error - {e}")

    print(f"\n{'='*60}")
    print(f"✅ Cleanup complete!")
    print(f"   Total non-shot labels removed: {total_removed}")
    print(f"   Files processed: {len(json_files)}")
    print(f"{'='*60}")

if __name__ == '__main__':
    training_dir = 'training_data'

    if not os.path.exists(training_dir):
        print(f"❌ Directory not found: {training_dir}")
        print("   Make sure you're running this from the project root")
        exit(1)

    print("="*60)
    print("🧹 Cleaning up non-shot labels from JSON files")
    print("="*60)
    print(f"Directory: {training_dir}\n")

    cleanup_json_files(training_dir)
