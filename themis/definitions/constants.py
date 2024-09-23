import os.path as osp
from pathlib import Path

PROJECT = "Themis"

# Paths
ROOT_PATH = Path(__file__).parent.parent.parent
TASK_PATH = osp.join(ROOT_PATH, "tasks")
TEST_PATH = osp.join(ROOT_PATH, "test")
SOURCE_PATH = osp.join(ROOT_PATH, "themis")
NOTEBOOK_PATH = osp.join(ROOT_PATH, "notebooks")
# Data
DATA_PATH = osp.join(ROOT_PATH, "data")
RAW_PATH = osp.join(DATA_PATH, "raw")
PROCESSED_PATH = osp.join(DATA_PATH, "processed")
LOGS_PATH = osp.join(DATA_PATH, "logs")
EXPERIMENTS_PATH = osp.join(LOGS_PATH, "experiments")
# Configs
CONFIG_PATH = osp.join(DATA_PATH, "config.yaml")
LOG_CONFIG_PATH = osp.join(DATA_PATH, "logging.yaml")
