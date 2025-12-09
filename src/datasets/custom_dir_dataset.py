import torch
import numpy as np
from pathlib import Path

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir=None, audio_dir=None, resynthesize=False, *args, **kwargs):
        self.resynthesize = resynthesize

        self.root_path = Path(data_dir).resolve()
        if not self.root_path.exists():
            raise ValueError(f"Given directory with utterances path does not exist: {self.root_path}")
        
        self.transcription_dir = self.root_path / "transcriptions"
        
        self.audio_dir = Path(audio_dir if audio_dir else self.root_path / 'wavs').resolve()

        index_data = self._scan_directory()

        super().__init__(index_data, *args, **kwargs)

    def _scan_directory(self):
        if not self.transcription_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.transcription_dir}")

        dataset_index = []
        
        txt_files = sorted(list(self.transcription_dir.glob("*.txt")))

        for txt_path in txt_files:
            file_id = txt_path.stem

            with open(txt_path, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()

            entry = {
                "id": file_id,
                "trans": transcription,
                "audio_path": self.audio_dir / f"{file_id}.wav",
                "mel_path": None
            }

            if not self.resynthesize:
                mel_path = self.transcription_dir / "mel" / f"{file_id}.npy"
                if mel_path.exists():
                    entry["mel_path"] = mel_path

            dataset_index.append(entry)

        return dataset_index

    def __getitem__(self, idx):
        entry = self._index[idx]
        
        item_id = entry["id"]
        transcription = entry["trans"]
        audio = None
        mel = None

        if self.resynthesize:
            audio = self.load_audio(str(entry["audio_path"])).squeeze()
            mel = self.make_mel(audio)
            
        else:
            if entry["mel_path"] and entry["mel_path"].exists():
                mel = torch.from_numpy(np.load(entry["mel_path"])).float().squeeze()
            else:
                if entry["audio_path"].exists():
                     audio = self.load_audio(str(entry["audio_path"])).squeeze()
            
            if entry["audio_path"].exists() and audio is None:
                audio = self.load_audio(str(entry["audio_path"])).squeeze()

        result = {
            "id": item_id,
            "trans": transcription,
            "mel": mel,
            "audio": audio
        }

        return self.preprocess_data(result)