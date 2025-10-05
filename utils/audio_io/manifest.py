"""
Audio file manifest creation and management.
Recursively scans directories for WAV files.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


class ManifestCreator:
    """Creates and manages audio file manifests."""
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize manifest creator.
        
        Args:
            num_workers: Number of parallel workers for file scanning
        """
        self.num_workers = num_workers
        
    def scan_directory(self, root_dir: Path, pattern: str = "*.wav") -> List[Path]:
        """
        Recursively scan directory for audio files.
        
        Args:
            root_dir: Root directory to scan
            pattern: File pattern to match
            
        Returns:
            List of audio file paths
        """
        root_dir = Path(root_dir)
        files = list(root_dir.rglob(pattern))
        return sorted(files)
        
    def get_wav_info(self, filepath: Path) -> Dict[str, Any]:
        """
        Extract WAV file information.
        
        Args:
            filepath: Path to WAV file
            
        Returns:
            Dictionary with file metadata
        """
        try:
            with wave.open(str(filepath), 'rb') as wav:
                info = {
                    'filepath': str(filepath),
                    'filename': filepath.name,
                    'sample_rate': wav.getframerate(),
                    'channels': wav.getnchannels(),
                    'duration': wav.getnframes() / wav.getframerate(),
                    'frames': wav.getnframes(),
                    'sample_width': wav.getsampwidth()
                }
            return info
        except Exception as e:
            return {
                'filepath': str(filepath),
                'filename': filepath.name,
                'error': str(e)
            }
            
    def create_manifest(self, 
                       data_dirs: List[Path], 
                       output_path: Path,
                       include_metadata: bool = True) -> pd.DataFrame:
        """
        Create manifest CSV file from audio directories.
        
        Args:
            data_dirs: List of directories to scan
            output_path: Output CSV file path
            include_metadata: Whether to include file metadata
            
        Returns:
            Manifest DataFrame
        """
        all_files = []
        for data_dir in data_dirs:
            files = self.scan_directory(data_dir)
            all_files.extend(files)
            
        if include_metadata:
            manifest_data = self._get_metadata_parallel(all_files)
        else:
            manifest_data = [{'filepath': str(f)} for f in all_files]
        
        # Convert to DataFrame
        df = pd.DataFrame(manifest_data)
        
        # Write to CSV
        self._write_manifest(manifest_data, output_path)
        
        return df
        
    def _get_metadata_parallel(self, files: List[Path]) -> List[Dict[str, Any]]:
        """
        Get metadata for files in parallel.
        
        Args:
            files: List of file paths
            
        Returns:
            List of metadata dictionaries
        """
        metadata = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.get_wav_info, f): f for f in files}
            
            for future in tqdm(as_completed(futures), total=len(files), 
                             desc="Scanning files"):
                result = future.result()
                metadata.append(result)
                
        return sorted(metadata, key=lambda x: x['filepath'])
        
    def _write_manifest(self, data: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Write manifest data to CSV.
        
        Args:
            data: Manifest data
            output_path: Output CSV path
        """
        if not data:
            return
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = list(data[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


def create_manifest(data_dirs: List[Path], 
                   output_path: Path,
                   num_workers: int = 4) -> pd.DataFrame:
    """
    Create audio file manifest.
    
    Args:
        data_dirs: Directories to scan
        output_path: Output CSV path
        num_workers: Number of parallel workers
        
    Returns:
        Manifest DataFrame
    """
    creator = ManifestCreator(num_workers)
    return creator.create_manifest(data_dirs, output_path)


def scan_audio_files(root_dir: Path,
                    extensions: Optional[List[str]] = None,
                    recursive: bool = True,
                    include_metadata: bool = True) -> pd.DataFrame:
    """
    Scan directory for audio files and return as DataFrame.
    This function is compatible with the main.py usage.
    
    Args:
        root_dir: Root directory to scan
        extensions: List of file extensions (e.g., ['.wav', '.flac'])
        recursive: Whether to scan recursively
        include_metadata: Whether to include file metadata
        
    Returns:
        DataFrame with audio file information
    """
    root_dir = Path(root_dir)
    
    if extensions is None:
        extensions = ['.wav']
    
    # Find all matching files
    all_files = []
    for ext in extensions:
        pattern = f"*{ext}" if not ext.startswith('.') else f"*{ext}"
        if recursive:
            files = list(root_dir.rglob(pattern))
        else:
            files = list(root_dir.glob(pattern))
        all_files.extend(files)
    
    all_files = sorted(set(all_files))  # Remove duplicates and sort
    
    if not all_files:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['filepath', 'filename', 'sample_rate', 
                                    'channels', 'duration', 'frames', 'sample_width'])
    
    # Get metadata if requested
    if include_metadata:
        creator = ManifestCreator()
        manifest_data = creator._get_metadata_parallel(all_files)
    else:
        manifest_data = [{'filepath': str(f), 'filename': f.name} for f in all_files]
    
    return pd.DataFrame(manifest_data)