import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """merges audio samples into one batch."""
    waveforms = [item["data_object"] for item in dataset_items]
    result_batch = {
        "data_object": pad_sequence(waveforms, batch_first=True),
        "lengths": torch.tensor([waveform.numel() for waveform in waveforms]),
        "labels": torch.tensor([item["labels"] for item in dataset_items]),
        "utterance_id": [item["utterance_id"] for item in dataset_items],
    }
    return result_batch
