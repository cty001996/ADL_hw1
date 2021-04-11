import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

from torch.utils.data import DataLoader
from model import SeqClassifier
import csv

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        for split, dataset in datasets.items()
    }
    # COMPLETE

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout,
                    args.bidirectional, len(intent2idx))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # COMPLETE

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # COMPLETE

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        size = len(dataloaders[TRAIN].dataset)
        loss_fn = torch.nn.CrossEntropyLoss()
        model.train()
        for batch_num, batch in enumerate(dataloaders[TRAIN]):
            encoded = batch["encoded"]
            label = batch["label"]
            
            if torch.cuda.is_available():
                encoded = encoded.cuda()
                label = label.cuda()
            pred = model(encoded)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_num % 100 == 0:
                loss, current = loss, batch_num * len(encoded)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        # COMPLETE
        # TODO: Evaluation loop - calculate accuracy and save model weights
        loss, correct = 0, 0
        size = len(dataloaders[DEV].dataset)
        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(dataloaders[DEV]):
                encoded = batch["encoded"]
                label = batch["label"]
                
                if torch.cuda.is_available():
                    encoded = encoded.cuda()
                    label = label.cuda()
                pred = model(encoded)
                loss += loss_fn(pred, label)
                pred_label = torch.argmax(pred, dim=1)
                correct += (pred_label == label).type(torch.float).sum()

        loss /= size
        accuracy = correct / size
        print(f"Dev Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    torch.save(model, args.ckpt_dir / "best.pt")
    # COMPLETE
    # TODO: Inference on test set

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
