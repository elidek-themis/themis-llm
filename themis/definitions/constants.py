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
HF_DATA_PATH = osp.join(SOURCE_PATH, "data")
DATA_PATH = osp.join(ROOT_PATH, "data")
CONFIG_PATH = osp.join(DATA_PATH, "conf")
TEMPLATES_PATH = osp.join(DATA_PATH, "templates")
RAW_PATH = osp.join(DATA_PATH, "raw")
LOGS_PATH = osp.join(DATA_PATH, "logs")
EXPERIMENTS_PATH = osp.join(DATA_PATH, "experiments")
