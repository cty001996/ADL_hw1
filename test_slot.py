import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqSlotDataset
from model import SeqSlot
from utils import Vocab

from torch.utils.data import DataLoader
import csv


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqSlotDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    '''
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()
'''
    model = torch.load(args.ckpt_path)
    model.eval()
    # load weights into model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
    # TODO: predict dataset
    pred_list = []
    id_list = []
    with torch.no_grad():
        for batch in dataloader:
            encoded = batch["encoded"]
            id = batch["id"]
            if torch.cuda.is_available():
                encoded = encoded.cuda()
            pred = model(encoded)
            pred = pred.permute(1,0,2)
            pred = pred.reshape(-1, pred.shape[-1])
            pred_tag = torch.argmax(pred, dim=1)
            pred_tag = " ".join(dataset.idx2tag(x.item()) for x in pred_tag)
            #print(pred_tag)
            pred_list.append(pred_tag)
            id_list.append(id[0])
    
    # COMPLETE

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        writer.writerow(['id', 'tags'])
        for i in range(len(id_list)):
            writer.writerow([id_list[i], pred_list[i]])
    # COMPLETE

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
