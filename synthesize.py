import os
import warnings
from pathlib import Path
import numpy as np
import torch
import torchaudio
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Local imports
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.utils import MelSpectrogram, MelSpectrogramConfig
from src.datasets.data_utils import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)


class AcousticPreprocessor:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.utils = None
        self.chunk_size = 100

    def _load_tacotron(self):
        if self.model is None:
            self.model = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub', 
                'nvidia_tacotron2', 
                model_math='fp16'
            ).to(self.device).eval()
            
            self.utils = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub', 
                'nvidia_tts_utils'
            )

    def generate_from_text(self, text: str) -> np.ndarray:
        self._load_tacotron()
        
        sequences, lengths = self.utils.prepare_input_sequence([text])
        seq_len = lengths.item()
        generated_mels = []
        
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        with torch.no_grad():
            for k in range(num_chunks):
                start_idx = self.chunk_size * k
                end_idx = self.chunk_size * (k + 1)
                
                chunk = sequences[:, start_idx:end_idx]
                chunk_len = torch.tensor([chunk.shape[1]], device=self.device)
                
                mel_chunk, _, _ = self.model.infer(chunk, chunk_len)
                generated_mels.append(mel_chunk.detach().cpu().squeeze())

        if generated_mels:
            return np.hstack(generated_mels)
        return None

    def process_directory(self, target_dir: str):
        target_path = Path(target_dir).absolute() / "transcriptions"
        output_mel_dir = target_path / "mel"

        if not target_path.exists():
            print(f"Directory {target_path} does not exist. Skipping acoustic processing.")
            return

        output_mel_dir.mkdir(parents=True, exist_ok=True)
        txt_files = sorted(list(target_path.glob("*.txt")))

        for txt_file in txt_files:
            file_id = txt_file.stem
            save_path = output_mel_dir / f"{file_id}.npy"
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                text_content = f.read()

            mel = self.generate_from_text(text_content)
            if mel is not None:
                np.save(str(save_path), mel)


class InferenceRunner:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.save_dir = ROOT_PATH / config.inferencer.save_path
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.writer = None
        if config.inferencer.log:
            project_conf = OmegaConf.to_container(config)
            self.writer = instantiate(config.writer, None, project_conf)

        self.mel_converter = MelSpectrogram(MelSpectrogramConfig())
        self.sample_rate = MelSpectrogramConfig().sr
        self.model = None

    def load_model(self):
        if self.model is None:
            model = instantiate(self.config.generator).to(self.device)
            checkpoint_path = self.config.inferencer.from_pretrained
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["gen_state_dict"])
            model.eval()
            self.model = model
        return self.model

    def synthesize_single(self, mel_numpy: np.ndarray, text: str, file_id: str = "query_result"):
        model = self.load_model()
        
        mel_tensor = torch.from_numpy(mel_numpy).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            generated_audio = model(mel_tensor)['fake'].detach().cpu() # [1, 1, T_audio]

        dummy_batch = {
            'id': [file_id],
            'real_mel': mel_tensor.cpu(),
            'text': [text],
            'real': [None]
        }
        
        self._save_and_log_batch(dummy_batch, generated_audio, current_step=0)

    def run(self, dataloader):
        model = self.load_model()
        step_counter = 0

        with torch.no_grad():
            for batch in dataloader:
                real_mels = batch['real_mel'].to(self.device)
                generated_audio = model(real_mels)['fake'].detach().cpu()
                step_counter = self._save_and_log_batch(
                    batch, generated_audio, step_counter
                )

    def _save_and_log_batch(self, batch, generated_audios, current_step):
        batch_ids = batch['id']
        
        for i, (audio_waveform, file_id) in enumerate(zip(generated_audios, batch_ids)):
            output_file = self.save_dir / f"{file_id}.wav"
            torchaudio.save(output_file, audio_waveform, self.sample_rate)

            if self.writer:
                self.writer.set_step(current_step)
                current_step += 1
                
                self._log_entry(
                    writer=self.writer,
                    fake_audio=audio_waveform,
                    real_audio=batch.get('real', [None])[i],
                    real_mel=batch['real_mel'][i],
                    text=batch.get('text', [""])[i],
                    file_id=file_id
                )
        return current_step

    def _log_entry(self, writer, fake_audio, real_audio, real_mel, text, file_id):
        writer.add_audio("fake_audio", fake_audio, self.sample_rate)
        if real_audio is not None:
            writer.add_audio("real_audio", real_audio, self.sample_rate)
            
        fake_mel = self.mel_converter(fake_audio).squeeze()
        writer.add_image("fake_mel", fake_mel)
        writer.add_image("real_mel", real_mel)
        writer.add_text("text", text)
        writer.add_text("id", file_id)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    set_random_seed(config.inferencer.seed)
    
    if config.inferencer.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.inferencer.device)
    
    pipeline = InferenceRunner(config, device)

    query = config.get("query")

    if query:
        preprocessor = AcousticPreprocessor(device)
        
        mel = preprocessor.generate_from_text(query)
        
        if mel is not None:
            pipeline.synthesize_single(mel, query, file_id="cli_query_result")
        else:
            print("Failed to generate MelSpectrogram from query.")
    else:
        if not config.datasets.test.resynthesize:
            preprocessor = AcousticPreprocessor(device)
            preprocessor.process_directory(config.datasets.test.data_dir)

        dataloaders, _ = get_dataloaders(config, device)
        test_loader = dataloaders['test']
        pipeline.run(test_loader)


if __name__ == "__main__":
    main()