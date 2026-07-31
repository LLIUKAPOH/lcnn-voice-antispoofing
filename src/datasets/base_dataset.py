from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """check that every dataset contain the required fields"""

    def __init__(self, index):
        self._assert_index_is_valid(index)
        self._index = index

    def __len__(self):
        return len(self._index)

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, "Each dataset item must include an audio path."
            assert "label" in entry, "Each dataset item must include a label"
