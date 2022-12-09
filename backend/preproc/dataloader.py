from logger import Logger
import pandas as pd
from typing import Callable

class Dataloader:
    '''
    Dataloader class
    ------
    Args:
        `fpath`(`str`): The path to the dataset file to read
    '''
    def __init__(self, fpath: str) -> None:
        # only supports xls & csv
        postfix = fpath.split('.')[-1]
        if not postfix in ['xlsx', 'csv']:
            Logger("The given file matches none of the supported format.").log('error')
            exit(1) # dev error, todo
        self.fpath = fpath
    
    def load(self) -> pd.DataFrame:
        # load job dispatcher
        if self.fpath.endswith('csv'): # prefix condition: file name ends with xlsx or csv
            Logger(f"CSV file {self.fpath} detected.").log('info')
            return self._load_csv()
        else:
            Logger(f"XLSX file {self.fpath} detected.").log('info')
            return self._load_xls()
    
    def _load_csv(self) -> pd.DataFrame:
        return self._load(pd.read_csv)

    def _load_xls(self) -> pd.DataFrame:
       return self._load(pd.read_excel)

    def _load(self, reader_function: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
        try:
            result: pd.DataFrame = reader_function(self.fpath)
        except ValueError:
            Logger(f"READ ERROR: cannot read file {self.fpath}.").log('error')
        Logger(f"Loaded dataset in {self.fpath}, head & first 2 rows:").log('ok')
        print(result.columns)
        print(result.head(2))
        return result
