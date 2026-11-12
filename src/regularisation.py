
"""
Regularisation configuration for Phase 2 Deep Learning.
Settings can be changed individually for ablation experiments.
"""

# Dropout
USE_DROPOUT = True
DROPOUT_RATE = 0.2

# Weight decay
USE_WEIGHT_DECAY = True
WEIGHT_DECAY = 1e-4

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

# Batch normalisation
USE_BATCH_NORM = False
