import torch
import torch.nn.functional as F

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.
    """
    
    PAD_MEL_VALUE = -11.5129251 
    
    batch_output = {}
    first_item = dataset_items[0]
    
    audio_tensors = [item['audio'] for item in dataset_items if item.get('audio') is not None]
    mel_tensors = [item['mel'] for item in dataset_items]
    
    if audio_tensors:
        max_audio_len = max(t.shape[0] for t in audio_tensors)
        
        padded_audio_list = [F.pad(a, (0, max_audio_len - a.shape[0])) for a in audio_tensors]
        batch_output['real'] = torch.stack(padded_audio_list).unsqueeze(1)
    
    max_mel_len = max(t.shape[1] for t in mel_tensors)
    
    padded_mel_list = [F.pad(m, (0, max_mel_len - m.shape[1]), value=PAD_MEL_VALUE) for m in mel_tensors]
    batch_output['real_mel'] = torch.stack(padded_mel_list)
    
    if first_item.get('id') is not None:
        batch_output['id'] = [item['id'] for item in dataset_items]
        
    if first_item.get('trans') is not None:
        batch_output['text'] = [item['trans'] for item in dataset_items] 
        
    return batch_output
