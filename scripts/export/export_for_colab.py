"""
Export Project for Google Colab

Creates a zip file containing only the files needed for training in Colab.
Excludes virtual environments, cache files, and local-only components.

Usage:
    python scripts/export_for_colab.py
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def should_include(path: Path, project_root: Path) -> bool:
    """Determine if a file should be included in the export."""
    rel_path = path.relative_to(project_root)
    rel_str = str(rel_path)
    
    # Exclude patterns
    exclude_patterns = [
        '.venv',
        '__pycache__',
        '.git',
        '.pytest_cache',
        'node_modules',
        '.ipynb_checkpoints',
        '*.pyc',
        '.DS_Store',
        'Thumbs.db',
        # Local-only folders
        'app/',  # Streamlit app (local only)
        'tests/',  # Tests (local only)
        'src/inference/',  # Inference (local only)
        'src/speech/',  # Speech (local only)
        'models/checkpoints/',  # Empty until trained
        'logs/',
        # Raw data (only need processed)
        'data/raw/',
    ]
    
    for pattern in exclude_patterns:
        if pattern.endswith('/'):
            if rel_str.startswith(pattern) or f'/{pattern}' in rel_str:
                return False
        elif pattern.startswith('*'):
            if rel_str.endswith(pattern[1:]):
                return False
        elif pattern in rel_str:
            return False
    
    return True


def export_for_colab():
    """Create a zip file for Colab upload."""
    project_root = get_project_root()
    
    # Output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"telugu-qa-system-colab-{timestamp}.zip"
    zip_path = project_root / zip_filename
    
    print("="*60)
    print("📦 Exporting Telugu QA System for Google Colab")
    print("="*60)
    
    files_added = 0
    total_size = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in project_root.rglob('*'):
            if file_path.is_file() and should_include(file_path, project_root):
                rel_path = file_path.relative_to(project_root)
                arcname = f"telugu-qa-system/{rel_path}"
                
                zipf.write(file_path, arcname)
                files_added += 1
                total_size += file_path.stat().st_size
                
                print(f"  + {rel_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Export complete!")
    print(f"{'='*60}")
    print(f"\nOutput: {zip_path}")
    print(f"Files: {files_added}")
    print(f"Size: {total_size / 1024 / 1024:.1f} MB")
    
    print(f"\n📋 Next Steps:")
    print(f"1. Open Google Colab")
    print(f"2. Upload {zip_filename}")
    print(f"3. Run the training notebook")
    
    return zip_path


if __name__ == "__main__":
    export_for_colab()
