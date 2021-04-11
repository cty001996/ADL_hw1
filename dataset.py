from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict: # -> ???
        # TODO: implement collate_fn
        text_batch = [sample["text"].split() for sample in samples]
        encoded_batch = self.vocab.encode_batch(text_batch)
        encoded_batch = torch.LongTensor(encoded_batch)
        if samples[0].get("intent") != None:
            label_batch = [self.label2idx(sample["intent"]) for sample in samples]
            label_batch = torch.LongTensor(label_batch)
        else:
            label_batch = None
        id_batch = [sample["id"] for sample in samples]
        return {"encoded": encoded_batch,
                "label": label_batch,
                "id": id_batch,
                }
        # COMPLETE

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqSlotDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        tag_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.tag_mapping = tag_mapping
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.tag_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict: # -> ???
        # TODO: implement collate_fn 
        samples = sorted(samples, key = lambda sample: len(sample["tokens"]), reverse=True)
        tokens_batch = [sample["tokens"] for sample in samples]
        lens = [len(sample["tokens"]) for sample in samples]
        encoded_batch = self.vocab.encode_batch(tokens_batch)
        encoded_batch = torch.LongTensor(encoded_batch).t()
        if samples[0].get("tags") != None:
            pad_len = lens[0]
            tag_batch = []
            for sample in samples:
                tag_batch.append ([self.tag2idx(tag) for tag in sample["tags"]]
                        + [self.tag2idx("O")] * (pad_len - len(sample["tags"])))
            tag_batch = torch.LongTensor(tag_batch).t()
        else:
            tag_batch = None
        id_batch = [sample["id"] for sample in samples]
        return {"encoded": encoded_batch,
                "tag": tag_batch,
                "id": id_batch,
                "lens": lens,
                }
        # COMPLETE

    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]
