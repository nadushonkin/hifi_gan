from pathlib import Path
import pandas as pd

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class LJSpeechDataset(BaseDataset):
    _CUTOFF_INDEX = 11791 # 0,9 * Total Clips (13100)

    def __init__(self, part, data_dir=None, *args, **kwargs):
        default_path = ROOT_PATH / "data" / "datasets" / "dla_dataset"
        self._data_dir = Path(data_dir or default_path).resolve()
        
        print(f"Loading dataset from: {self._data_dir}")
        
        index_data = self._load_and_process_index(part)
        super().__init__(index_data, *args, **kwargs)

    def _load_and_process_index(self, part):
        csv_path = self._data_dir / "metadata.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        df = pd.read_csv(
            csv_path, 
            sep='|', 
            header=None, 
            names=['id', 'trans', 'unused'], 
            usecols=['id', 'trans']
        )

        if part == "train":
            df = df.iloc[:self._CUTOFF_INDEX]
        elif part == "test":
            df = df.iloc[self._CUTOFF_INDEX:]
        
        wav_dir_str = str(self._data_dir / "wavs")
        df["path"] = wav_dir_str + "/" + df["id"] + ".wav"

        return df.to_dict("records")