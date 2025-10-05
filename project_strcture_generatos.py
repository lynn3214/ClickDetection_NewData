#!/usr/bin/env python3
"""
Dolphin Click Detection Project Structure Generator
生成完整的项目目录结构和模块定义
"""

import os
from pathlib import Path


def create_project_structure():
    """创建完整的项目目录结构"""
    
    structure = {
        "data": {
            "raw": {
                "training_sources": {},
                "noise": {},
                "wav": {},
                "mat": {}
            },
            "test_raw": {},
            "resampled_training": {},
            "test_resampled": {},
            "noise_resampled": {},
            "snippets": {
                "events": {},
                "trains": {}
            }
        },
        "preprocessing": {
            "scan_manifest": {},
            "convert_and_filter": {}
        },
        "detection": {
            "rules": {},
            "candidate_finder": {},
            "segmenter": {},
            "features_event": {},
            "train_builder": {},
            "fusion": {},
            "export": {}
        },
        "training": {
            "dataset": {},
            "augment": {},
            "train": {},
            "eval": {}
        },
        "models": {
            "cnn1d": {},
            "checkpoints": {}
        },
        "utils": {
            "audio_io": {},
            "dsp": {},
            "config": {},
            "logging": {},
            "metrics": {}
        },
        "configs": {},
        "reports": {
            "qa_plots": {},
            "metrics": {}
        },
        "data_scripts": {}
    }
    
    def create_dirs(base_path: Path, struct: dict):
        """递归创建目录结构"""
        for name, children in struct.items():
            dir_path = base_path / name
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # 创建__init__.py（除了data和configs）
            if name not in ["data", "configs", "reports", "data_scripts"]:
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text('"""{}"""\n'.format(name.replace('_', ' ').title()))
            
            if children:
                create_dirs(dir_path, children)
    
    project_root = Path.cwd()
    create_dirs(project_root, structure)
    print("✅ 项目目录结构创建完成")


if __name__ == "__main__":
    create_project_structure()