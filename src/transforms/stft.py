import torch
from torch import nn
from torch.nn import functional as functional


class LogSTFT(nn.Module):
    """convert padded waveforms to raw log-power STFT features."""

    def __init__(
        self,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        frame_count=None,
        window_type="hann",
        center=True,
        random_crop=False,
        eps=1e-6,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.frame_count = frame_count
        self.center = center
        self.random_crop = random_crop
        self.eps = eps

        if window_type == "hann":
            window = torch.hann_window(win_length)
        elif window_type == "blackman":
            window = torch.blackman_window(win_length)
        else:
            raise ValueError(f"Unknown window type: {window_type}")

        self.register_buffer("window", window, persistent=False)

    def _valid_frame_count(self, waveform_length):
        waveform_length = max(int(waveform_length), self.n_fft)
        if self.center:
            return 1 + waveform_length // self.hop_length
        return 1 + (waveform_length - self.n_fft) // self.hop_length

    def _fit_frame_count(self, spectrogram):
        if self.frame_count is None:
            return spectrogram

        available_frames = spectrogram.size(-1)
        if available_frames >= self.frame_count:
            if self.random_crop and available_frames > self.frame_count:
                start = torch.randint(available_frames - self.frame_count + 1, (1,)).item()
            else:
                start = 0
            return spectrogram[..., start : start + self.frame_count]

        repeats = (self.frame_count + available_frames - 1) // available_frames
        return spectrogram.repeat(1, repeats)[..., : self.frame_count]

    def forward(self, x, lengths=None):
        """
        Args:
            x: Padded mono waveforms of shape ``(batch, samples)``.
            lengths: Original waveform lengths before collate padding.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms with shape (batch, samples), got {tuple(x.shape)}")

        
        if x.size(-1) < self.n_fft:
            x = functional.pad(x, (0, self.n_fft - x.size(-1)))

        spectrum = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        log_power = torch.log(spectrum.abs().square() + self.eps)

        if lengths is None:
            frame_lengths = [log_power.size(-1)] * log_power.size(0)
        else:
            if len(lengths) != log_power.size(0):
                raise ValueError("Waveform lengths must contain one value per batch item.")
            frame_lengths = [self._valid_frame_count(length) for length in lengths]

        features = []
        for feature, frame_length in zip(log_power, frame_lengths):
            features.append(self._fit_frame_count(feature[..., :frame_length]))
        return torch.stack(features, dim=0).unsqueeze(1)
