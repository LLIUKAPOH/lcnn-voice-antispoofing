from pathlib import Path

import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVspoof2019Dataset(BaseDataset):


    def __init__(self, root_dir, partition, *args, **kwargs):
        self.root_dir = Path(root_dir)
        self.partition = partition
        self.la_root = self._find_la_root()
        index = self._create_index()
        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        """loads ASVspoof 2019 LA protocol and FLAC audio"""
        audio_dir = self.la_root / f"ASVspoof2019_LA_{self.partition}" / "flac"
        protocol_path = self._find_protocol()
        index = []
        with protocol_path.open() as protocol_file:
            for line in protocol_file:
                speaker_id, utterance_id, _, attack_id, label = line.split()
                index.append(
                    {
                        "path": str(audio_dir / f"{utterance_id}.flac"),
                        "label": int(label == "bonafide"),
                        "utterance_id": utterance_id,
                        "speaker_id": speaker_id,
                        "attack_id": attack_id,
                    }
                )
        return index

    def __getitem__(self, ind):
        """returns item as dict basically"""
        item = self._index[ind]
        waveform, sample_rate = torchaudio.load(item["path"])
        """given dataset contains only one chanel (mono chanel) and 16kHz"""
        waveform = waveform.mean(dim=0)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return {
            "data_object": waveform,
            "labels": item["label"],
            "utterance_id": item["utterance_id"],
        }

    def _find_la_root(self):
        candidates = [self.root_dir, self.root_dir / "LA"]
        candidates.extend(
            path / "LA" for path in self.root_dir.iterdir() if path.is_dir()
        )
        for candidate in candidates:
            audio_dir = candidate / f"ASVspoof2019_LA_{self.partition}" / "flac"
            if audio_dir.is_dir():
                return candidate
        raise FileNotFoundError(f"LA partition not found in {self.root_dir}.")

    def _find_protocol(self):
        protocol_dir = self.la_root / "ASVspoof2019_LA_cm_protocols"
        candidates = sorted(
            protocol_dir.glob(f"ASVspoof2019.LA.cm.{self.partition}*.txt")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one CM protocol for '{self.partition}' in {self.root_dir}."
            )
        return candidates[0]
