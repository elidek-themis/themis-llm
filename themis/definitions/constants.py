import os.path as osp
from pathlib import Path

PROJECT = "Themis"


def get_project_root() -> Path:
    """Singleton instance"""
    return Path(__file__).parent.parent.parent


# Paths
ROOT_PATH = get_project_root()
DATA_PATH = osp.join(ROOT_PATH, "data")
LOGS_PATH = osp.join(DATA_PATH, "logs")
EXPERIMENTS_PATH = osp.join(LOGS_PATH, "experiments")
CONFIG_PATH = osp.join(DATA_PATH, "config.yaml")
LOG_CONFIG_PATH = osp.join(DATA_PATH, "log_config.json")
TASK_PATH = osp.join(ROOT_PATH, "tasks")
TEST_PATH = osp.join(ROOT_PATH, "test")
SOURCE_PATH = osp.join(ROOT_PATH, "themis")
NOTEBOOK_PATH = osp.join(ROOT_PATH, "notebooks")
