#!/usr/bin/env python3
"""
Quick 1-epoch training to generate checkpoint for visualization
"""

import sys
from pathlib import Path

# Modify config to run just 1 epoch
sys.path.append(str(Path(__file__).parent.parent))

# Import and modify the main function
from experiments_fgadr.train_multiclass_segmentation import *

if __name__ == "__main__":
    # Override num_epochs to 1
    import importlib
    import experiments_fgadr.train_multiclass_segmentation as train_module
    
    # Run with 1 epoch
    original_main = train_module.main
    
    def quick_main():
        # Call original main but we'll modify config inside
        original_main()
    
    # Monkey patch to run 1 epoch
    config_backup = None
    
    # Just run the main with env var
    import os
    os.environ['QUICK_TEST'] = '1'
    
    main()
