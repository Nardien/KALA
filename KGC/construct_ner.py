"""
Knowledge Graph Construction based on the context
The function can be embedded into main run file,
or it can be executed isolately. 
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

from transformers import AutoConfig, AutoTokenizer, default_data_collator

from model import BertForDocRED

from datasets import load_dataset, ClassLabel

def deduplicate(seq):
    unique = set()
    for d in seq:
        t = tuple(sorted(d.items(), key=lambda x: x[0]))
        unique.add(t)
    return [dict(x) for x in unique]

def norm_mask(input_mask):
    output_mask = np.zeros(input_mask.shape)
    for i in range(len(input_mask)):
        if not np.all(input_mask[i] == 0):
            output_mask[i] = input_mask[i] / sum(input_mask[i])
    return output_mask

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def char_to_token_idx(text, tokens, entity, length):
    start_char = entity["start"]
    end_char = entity["end"] - 1

    while end_char >= len(text) or is_whitespace(text[end_char]):
        end_char -= 1

    start_idx = tokens.char_to_token(start_char)
    end_idx = tokens.char_to_token(end_char)

    if start_idx is not None:
        start_idx += length
    if end_idx is not None:
        end_idx += length
    return start_idx, end_idx

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

    """ (S) Load Dataset for NER """

    raw_datasets = load_dataset(args.domain, cache_dir='../dataset')

    with open(args.train_file, 'r') as infile:
        doc_id_to_entities = json.load(infile)

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = f"{args.task_name}_tags" if f"{args.task_name}_tags" in column_names else column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Preprocessing the raw_datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = raw_datasets[args.fold]
    dataset = dataset.map(
        tokenize_and_align_labels, batched=True
    )

    def align_entities(example):

        tokens = example['tokens']
        doc_text = " ".join(tokens)

        tokenized_inputs = tokenizer(
            doc_text,
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # is_split_into_words=True,
        )

        assert tokenized_inputs.input_ids == example['input_ids']

        doc_id = example['id']
        entities = []
        if doc_id in doc_id_to_entities.keys():
            entities = doc_id_to_entities[doc_id]

        wikidata_ids = []
        ent_pos = np.zeros((args.max_ent_cnt, args.max_length), dtype='int32')
        ent_mask = [0] * args.max_ent_cnt
        ent_idx = 0

        for entity in entities:

            if ent_idx >= args.max_ent_cnt:
                break

            start_idx, end_idx = char_to_token_idx(
                doc_text, tokenized_inputs, entity, 0
            )
            if start_idx is None or end_idx is None:
                continue

            if start_idx >= args.max_length or end_idx >= args.max_length:
                continue

            span = (tokenized_inputs.input_ids)[start_idx:end_idx+1]
            pred = tokenizer.decode(span)
            ground_truth = tokenizer.decode(tokenizer.encode(entity["text"], add_special_tokens=False))
        
            if ground_truth != pred:
                # print(f"Entity Mismatch GT:{ground_truth} vs Found:{pred}")
                continue

            wikidata_id = entity['id']
            wikidata_ids.append(wikidata_id)

            for tok_idx in range(start_idx, end_idx+1):
                ent_pos[ent_idx][tok_idx] = 1
            ent_mask[ent_idx] = 1
            ent_idx += 1

        ent_pos = norm_mask(ent_pos)
        ent_len = len(wikidata_ids)

        label_mask = np.zeros((args.max_ent_cnt, args.max_ent_cnt), dtype='bool')
        label_mask[:ent_len, :ent_len] = 1
        for ent in range(ent_len):
            label_mask[ent][ent] = 0
        for ent in range(ent_len):
            if np.all(ent_mask[ent] == 0):
                label_mask[ent, :] = 0
                label_mask[:, ent] = 0

        example['ent_pos'] = ent_pos
        example['ent_mask'] = ent_mask
        example['label_mask'] = label_mask
        example['wikidata_ids'] = wikidata_ids

        return example

    dataset = dataset.map(
        align_entities
    )

    """ (E) Load Dataset for NER """

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value, dtype=torch.long):
            if isinstance(target, str):
                tensors = [torch.tensor(o[target], dtype=dtype) for o in batch]
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
        ret["feature_indices"] = torch.tensor([int(o['id']) for o in batch], dtype=torch.long)
        return ret

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )

    entity_embeddings_memory = dict()
    threshold = args.threshold

    doc_id_to_triplets = {}
    num_triplets = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Relation Extraction")):
        inputs = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs[0]
        ent_reps = outputs[1]

        """ Iterate over batch instances """
        for i, feature_index in enumerate(batch["feature_indices"]):
            feature = dataset[feature_index.item()]

            assert int(feature['id']) == feature_index.item()

            pred = preds[i]
            wikidata_ids = feature['wikidata_ids']
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

                    # if pred[h][t] == 0:
                    #     continue
                    relation = reverse_label_map[label]
                    triplets.append(
                        {
                            'h': wikidata_ids[h],
                            'r': relation,
                            't': wikidata_ids[t],
                        }
                    )

            doc_id_to_triplets[feature['id']] = deduplicate(triplets)
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

    doc_id_to_features = {}
    for key in doc_id_to_entities:
        doc_id_to_features[key] = {
            'entities': doc_id_to_entities[key],
            'triplets': doc_id_to_triplets[key]
        }

    new_filename = os.path.join(args.output_dir, f"{args.fold}_{args.threshold}.json")
    with open(new_filename, 'w') as outfile:
        json.dump(doc_id_to_features, outfile)

    if args.fold == "train":
        with open(os.path.join(args.output_dir, f"train_entity_embeddings_{args.threshold}.pkl"), 'wb+') as f:
            pickle.dump(entity_embeddings_memory, f)

    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process lines of text corpus into Knowledge Graph")
    parser.add_argument("--train_file", type=str,
                        default="/v5/jinheon/code/KALA_QA/dataset/")
    parser.add_argument("--domain", type=str, default="conll2003",
                        choices=['conll2003', 'wnut_17', 'ncbi_disease'])
    parser.add_argument("--fold", type=str, default="train", 
                        choices=['train', 'validation', 'test'])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max_ent_cnt", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=True,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )

    args = parser.parse_args()

    args.train_file = os.path.join(args.train_file, args.domain, f"{args.fold}_entity.json")
    args.output_dir = f"./TASK/{args.domain}/results"
    print(args.train_file)
    print(args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    construct(args)