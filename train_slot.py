import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqSlotDataset
from utils import Vocab

from torch.utils.data import DataLoader
from model import SeqSlot
import csv

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqSlotDataset] = {
        split: SeqSlotDataset(split_data, vocab, tag2idx, args.max_len)
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
    model = SeqSlot(embeddings, args.hidden_size, args.num_layers, args.dropout,
                    args.bidirectional, datasets[TRAIN].num_classes)
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
            tag = batch["tag"]
            lens = batch["lens"]
            if torch.cuda.is_available():
                encoded = encoded.cuda()
                tag = tag.cuda()
            pred = model(encoded, lens)
            pred = pred.view(-1, pred.shape[-1])
            tag = tag.reshape(-1)
            loss = loss_fn(pred, tag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_num % 50 == 0:
                loss, current = loss, batch_num * len(encoded)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        # COMPLETE
        # TODO: Evaluation loop - calculate accuracy and save model weights
        loss, size = 0, 0
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(dataloaders[DEV]):
                encoded = batch["encoded"]
                tag = batch["tag"]
                lens = batch["lens"]
                if torch.cuda.is_available():
                    encoded = encoded.cuda()
                    tag = tag.cuda()
                pred = model(encoded, lens)
                pred = pred.view(-1, pred.shape[-1])
                tag = tag.reshape(-1)
                loss += loss_fn(pred, tag)
                pred_tag = torch.argmax(pred, dim=1)
                pred_tag = pred_tag.view(-1, len(encoded))
                tag = tag.view(-1, len(encoded))
                
                size += len(encoded)
                tran_pred = pred_tag.t()
                tran_true = tag.t()
                tran_pred = [list(map(datasets[DEV].idx2tag, (tran_pred[i][:lens[i]]).tolist()))
                        for i in range(len(tran_pred))]
                tran_true = [list(map(datasets[DEV].idx2tag, (tran_true[i][:lens[i]]).tolist()))
                        for i in range(len(tran_true))]
                for i in range(len(tran_pred)):
                    y_pred.append(tran_pred[i])
                    y_true.append(tran_true[i])

        report = classification_report(y_pred, y_true, mode='strict', scheme=IOB2)
        print(report)
        join_correct = correct = 0
        join_count = count = 0
        for i in range(len(y_pred)):
            join = True
            for j in range(len(y_pred[i])):
                count += 1
                if y_pred[i][j] == y_true[i][j]:
                    correct += 1
                else:
                    join = False
            join_count += 1
            if join:
                join_correct += 1
        accuracy = correct / count
        join_ac = join_correct / join_count
        loss /= size
        print(f"Dev Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} JoinAC: {(100*join_ac):>0.1f}% \n")
    torch.save(model, args.ckpt_dir / "best.pt")
    # COMPLETE
    # TODO: Inference on test set

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
