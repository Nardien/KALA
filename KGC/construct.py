"""
Knowledge Graph Construction
This code needs fine-tuned language model for Relation Extraction
"""
import argparse
import os
import numpy as np
import random
import json
from tqdm import tqdm
import pickle
from collections import Counter

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoConfig, AutoTokenizer

from model import BertForDocRED
from dataset import read_dataset, convert_examples_to_features, find_data

def deduplicate(seq):
    unique = set()
    for d in seq:
        t = tuple(sorted(d.items(), key=lambda x: x[0]))
        unique.add(t)
    return [dict(x) for x in unique]

def construct(args):
    args.device = torch.device('cuda')

    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with open("../label_map.json", 'r') as f:
        label_map = json.load(f)

    reverse_label_map = {v:k for k, v in label_map.items()}

    config = AutoConfig.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForDocRED.from_pretrained("bert-base-uncased",
                                          config=config,
                                          num_labels=len(label_map.keys()),
                                          max_ent_cnt=args.max_ent_cnt,)
    model.load_state_dict(
        torch.load("../RE_checkpoint/model.pth")
    )
    model.to(args.device)
    model.eval()

    dataset = read_dataset(args)
    features = convert_examples_to_features(dataset, tokenizer, max_length=args.max_length, max_ent_cnt=args.max_ent_cnt)

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value, dtype=torch.long):
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=dtype) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=dtype) for o in target]
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            input_ids=create_padded_sequence("input_ids", tokenizer.pad_token_id),
            attention_mask=create_padded_sequence("attention_mask", 0),
            token_type_ids=create_padded_sequence("token_type_ids", 0),
            ent_pos=create_padded_sequence("ent_pos", 0, torch.float),
            ent_mask=create_padded_sequence("ent_mask", 0, torch.bool),
            label_mask=create_padded_sequence("label_mask", 0, torch.bool)
        )
        ret["feature_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)
        return ret

    sampler = SequentialSampler(features)
    dataloader = DataLoader(
        list(enumerate(features)), sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn
    )

    entity_embeddings_memory = dict()
    new_dataset = []
    threshold = args.threshold

    num_triplets = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Relation Extraction")):
        inputs = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs[0]
        ent_reps = outputs[1]

        """ Iterate over batch instances """
        for i, feature_index in enumerate(batch["feature_indices"]):
            feature = features[feature_index.item()]
            pred = preds[i]
            wikidata_ids = feature.wikidata_ids
            triplets = []

            for h in range(len(wikidata_ids)):
                for t in range(len(wikidata_ids)):
                    if h == t or wikidata_ids[h] == wikidata_ids[t]:
                        continue

                    if pred[h][t].argmax().item() == 0:
                        if pred[h][t][1:].max().item() < threshold:
                            continue
                        else:
                            label = pred[h][t][1:].argmax().item() + 1
                    else:
                        label = pred[h][t].argmax().item()

                    relation = reverse_label_map[label]
                    triplets.append(
                        {
                            'h': wikidata_ids[h],
                            'r': relation,
                            't': wikidata_ids[t],
                        }
                    )
            qa = find_data(dataset, feature.qid)
            if "triplets" in qa.keys():
                qa["triplets"] += triplets
                qa["triplets"] = deduplicate(qa["triplets"])
            else:
                qa["triplets"] = deduplicate(triplets)
           
            num_triplets += len(triplets)
        
            # Cache Entity Embedding
            ent_rep = ent_reps[i]
            for j, wikidata_id in enumerate(wikidata_ids):
                emb = ent_rep[j].cpu().numpy()
                if wikidata_id in entity_embeddings_memory.keys():
                    # Moving Average
                    entity_embeddings_memory[wikidata_id] = args.alpha * entity_embeddings_memory[wikidata_id] + (1 - args.alpha) * emb
                else:
                    entity_embeddings_memory[wikidata_id] = emb

    print(f"Average Number of Triplets: {num_triplets / len(dataset)}")

    new_filename = os.path.join(args.output_dir, f"{args.fold}.json")
    with open(new_filename, 'w') as outfile:
        json.dump({'data':dataset}, outfile)

    if args.fold == "train":
        with open(os.path.join(args.output_dir, "train_entity_embeddings.pkl"), 'wb+') as f:
            pickle.dump(entity_embeddings_memory, f)

    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process lines of text corpus into Knowledge Graph")
    parser.add_argument("--train_file", type=str,
                        default="../dataset")
    parser.add_argument("--domain", type=str, default="NewsQA")
    parser.add_argument("--fold", type=str, default="train", choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max_ent_cnt", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    args.train_file = os.path.join(args.train_file, args.domain, f"{args.fold}_entity.json")
    args.output_dir = f"./TASK/{args.domain}"
    print(args.train_file)
    print(args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    construct(args)
