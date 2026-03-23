# This file should contain small generic helpers, not model-specific logic.
# seed_everything()
# timer()
# memory_cleanup()
# save_pickle()
# load_pickle()

import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)