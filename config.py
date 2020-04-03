
from pathlib import Path 
import os


class Config(object):
    """
    Configuration for set up. 
    """

    local_path = Path("./local_files/")


def environment_setup():
    """
    Generate the folder structure needed for project.
    """
    if not os.path.exists(Config.local_path):
        os.makedirs(Config.local_path)
    if not os.path.exists(Config.local_path / "data"):
        os.makedirs(Config.local_path / "data")
    if not os.path.exists(Config.local_path / "figures"):
        os.makedirs(Config.local_path / "figures")

