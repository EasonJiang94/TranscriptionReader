import pandas as pd
from loguru import logger
import re
import os
import subprocess
class DataLoader:
    DES_STR = "description"
    TRAN_STR = "transcription"
    def __init__(self, path):
        self.path = path
        self._load()
        
    def _load(self):
        logger.info(f"Loading dataset : {self.path}")
        self.data = pd.read_csv(self.path)
        logger.debug(f"{self.data.columns}")
        pass

    def show(self, n=10, col=None):
        for idx, row in self.data.iterrows():
            if idx > n:
                break
            if col in self.data.columns:
                logger.info(f"{row[col]}")
            else:
                logger.debug(f"{row}")
    
    @property
    def columns(self):
        return self.data.columns


if __name__ == "__main__":
    path = "../../data/mtsamples.csv"
    dl = DataLoader(path)
    dl.show(100, col="transcription")
