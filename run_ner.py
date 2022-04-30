#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import json
import pickle
import numpy as np

import datasets
import torch
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from utils import char_to_token_idx

from collections import Counter

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

with open("./label_map.json") as f:
    label_map = json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

    # Extend
    parser.add_argument("--data_dir", type=str, default="./KGC/TASK/Biomedical/results")
    parser.add_argument("--threshold", type=float, default=0.05)

    parser.add_argument("--hidden_size", type=int, default=768)

    parser.add_argument("--kala_learning_rate", type=float, default=3e-5)
    parser.add_argument("--loc_layer", type=str, default="9,11")
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    assert args.dataset_name in ['conll2003', 'wnut_17', 'ncbi_disease']
    args.domain = args.dataset_name
    args.data_dir = args.data_dir.replace("Biomedical", args.domain)

    args.loc_layer = [int(x) for x in args.loc_layer.split(',')]

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def load_entity_embeddings_memory(args):
    memory_path = os.path.join(args.data_dir, f'train_entity_embeddings_{args.threshold}.pkl')

    with open(memory_path, 'rb') as infile:
        entity_embeddings_memory = pickle.load(infile)

    entity_filename = f"train_entity.json"
    entity_filename = os.path.join('./dataset', args.domain, entity_filename)

    wikidata_to_memory_map = dict()
    entity_embeddings = []

    # Note that "0" indicates "Zero Embedding"
    for idx, (key, value) in enumerate(entity_embeddings_memory.items()):
        wikidata_to_memory_map[key] = len(entity_embeddings) + 1
        entity_embeddings.append(value)

    entity_embeddings = torch.from_numpy(np.stack(entity_embeddings, axis=0))

    args.entity_embed_size = entity_embeddings.shape[-1]
    args.wikidata_to_memory_map = wikidata_to_memory_map

    entity_embeddings = torch.cat([torch.zeros(1, entity_embeddings.shape[-1]), entity_embeddings], dim=0)
    return entity_embeddings, wikidata_to_memory_map

def initialize_model(args, entity_embeddings):

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=args.num_labels, cache_dir='./dataset')
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels, cache_dir='./dataset')
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True, cache_dir='./dataset')
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, cache_dir='./dataset')
    args.tokenizer = tokenizer

    from models.KALA.modeling_bert import BertForNER
    if args.model_name_or_path:
        model = BertForNER.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            args=args,
            entity_embeddings=entity_embeddings,
            cache_dir='./dataset/'
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForTokenClassification.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def save_model(args, model):        
    ckpt_file = os.path.join(args.output_dir, "ckpt.pt")
    ckpt = {"args": args, "state_dict": model.state_dict()}
    torch.save(ckpt, ckpt_file)
    logger.info("SAVE MODEL!")

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name,
                                    cache_dir='./dataset')
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    with open(
            os.path.join(args.data_dir, f'train_{args.threshold}.json'), 'r'
        ) as infile:
        train_doc_id_to_features = json.load(infile)
    with open(
            os.path.join(args.data_dir, f'validation_{args.threshold}.json'), 'r'
        ) as infile:
        validation_doc_id_to_features = json.load(infile)
    with open(
            os.path.join(args.data_dir, f'test_{args.threshold}.json'), 'r'
        ) as infile:
        test_doc_id_to_features = json.load(infile)

    entity_embeddings, wikidata_to_memory_map = load_entity_embeddings_memory(args)

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
    else:
        label_column_name = column_names[1]

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
    
    args.num_labels = num_labels

    device = args.device = accelerator.device
    model, tokenizer = initialize_model(args, entity_embeddings)

    # Preprocessing the datasets.
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

    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            # remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    test_dataset = processed_raw_datasets["test"]

    def align_entities_and_graphs(example, **fn_kwargs):

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
        features = fn_kwargs['doc_id_to_features'][doc_id]

        example['entities'] = features['entities']
        example['triplets'] = features['triplets']

        wikidata_ids = []
        ent_pos = [-1] * args.max_length
        
        for entity in example['entities']:

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

            if wikidata_id in wikidata_ids:
                node_idx = wikidata_ids.index(wikidata_id)
            else:
                node_idx = len(wikidata_ids)
                wikidata_ids.append(wikidata_id)

            for tok_idx in range(start_idx, end_idx+1):
                ent_pos[tok_idx] = node_idx

        local_wikidata_ids = []
        global_to_local = {}
        local_ent_pos = []
        idx = 0
        for pos in list(set(ent_pos)):
            if pos < 0: continue
            local_wikidata_ids.append(wikidata_ids[pos])
            global_to_local[pos] = idx
            idx += 1
        for pos in ent_pos:
            local_ent_pos.append(global_to_local.get(pos, -1))
        assert max(local_ent_pos) + 1 == len(local_wikidata_ids)

        wikidata_ids = example['wikidata_ids'] = local_wikidata_ids
        ent_pos = example['ent_pos'] = local_ent_pos

        assert len(example['input_ids']) == args.max_length
        assert len(example['attention_mask']) == args.max_length
        assert len(example['ent_pos']) == args.max_length

        edge_index = []
        edge_attr = []
        local_indicator = [1 for _ in range(len(wikidata_ids))]
        if len(example['triplets']) > 0:
            for triplet in example['triplets']:
                if triplet['t'] in wikidata_ids and triplet['h'] not in wikidata_ids:
                    wikidata_ids.append(triplet['h'])
                    local_indicator.append(0)
                if triplet['h'] in wikidata_ids and triplet['t'] not in wikidata_ids:
                    wikidata_ids.append(triplet['t'])
                    local_indicator.append(0)
                if triplet['h'] not in wikidata_ids or triplet['t'] not in wikidata_ids:
                    continue

                edge = [wikidata_ids.index(triplet['h']), wikidata_ids.index(triplet['t'])]
                if edge not in edge_index:
                    edge_index.append(edge)
                    edge_attr.append(label_map[triplet['r']] - 1) # Shift due to "0 = no relation"

        example['edge_index'] = edge_index
        example['edge_attr'] = edge_attr
        example['local_indicator'] = local_indicator

        return example

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            align_entities_and_graphs,
            fn_kwargs={
                'doc_id_to_features': train_doc_id_to_features
            },
            desc="Running alignment of entity and graph on dataset",
        )
        eval_dataset = eval_dataset.map(
            align_entities_and_graphs,
            fn_kwargs={
                'doc_id_to_features': validation_doc_id_to_features
            },
            desc="Running alignment of entity and graph on dataset",
        )
        test_dataset = test_dataset.map(
            align_entities_and_graphs,
            fn_kwargs={
                'doc_id_to_features': test_doc_id_to_features
            },
            desc="Running alignment of entity and graph on dataset",
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value, dtype=torch.long):
            if isinstance(target, str):
                tensors = [torch.tensor(o[target], dtype=dtype) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=dtype) for o in target]
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        def retrieve(key):
            if key in wikidata_to_memory_map.keys():
                return wikidata_to_memory_map[key]
            else:
                return 0

        """ convert to torch_geometric batch type """
        batch_nodes = []
        batch_edge_index = []
        batch_edge_attr = []
        graph_batch = []
        batch_local_indicator = []
        for batch_idx, item in enumerate(batch):
            nodes = [retrieve(node) for node in item['wikidata_ids']]
            edge_index = [[len(graph_batch) + edge[0], len(graph_batch) + edge[1]] for edge in item['edge_index']]
            
            # Reverse
            rev_edge_index = []
            rev_edge_attr = []
            for edge, edge_attr in zip(item['edge_index'], item['edge_attr']):
                rev_edge = [len(graph_batch) + edge[1], len(graph_batch) + edge[0]]
                if rev_edge in edge_index:
                    continue
                rev_edge_index.append(rev_edge)
                rev_edge_attr.append(edge_attr)
            
            graph_batch += [batch_idx] * len(nodes)
            batch_nodes += nodes
            batch_edge_index += edge_index
            batch_edge_attr += item['edge_attr']
            
            batch_edge_index += rev_edge_index
            batch_edge_attr += rev_edge_attr
            batch_local_indicator += item['local_indicator']

        ret = dict(
            input_ids=create_padded_sequence("input_ids", args.tokenizer.pad_token_id),
            attention_mask=create_padded_sequence("attention_mask", 0),
            mention_positions=create_padded_sequence("ent_pos", -1),
            labels=create_padded_sequence("labels", -100),
            nodes=torch.tensor(batch_nodes, dtype=torch.long),
            edge_index=torch.tensor(batch_edge_index, dtype=torch.long).t().reshape(2, -1),
            edge_attr=torch.tensor(batch_edge_attr, dtype=torch.long),
            graph_batch=torch.tensor(graph_batch, dtype=torch.long),
            local_indicator=torch.tensor(batch_local_indicator, dtype=torch.long)
        )
        ret["feature_indices"] = torch.tensor([int(o['id']) for o in batch], dtype=torch.long)

        if args.model_name_or_path != "roberta-base":
            ret["token_type_ids"] = create_padded_sequence("token_type_ids", 0)

        return ret
        
    # # DataLoaders creation:
    # if args.pad_to_max_length:
    #     # If padding was already done ot max length, we use the default data collator that will just convert everything
    #     # to tensors.
    #     data_collator = default_data_collator
    # else:
    #     # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
    #     # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
    #     # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    #     data_collator = DataCollatorForTokenClassification(
    #         tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    #     )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metrics
    metric = load_metric("seqeval", cache_dir='./dataset')

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_f1 = 0
    best_epoch = 0
    save_model(args, model)

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
            outputs = model(**inputs)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            preds, refs = get_labels(predictions_gathered, labels_gathered)
            metric.add_batch(
                predictions=preds,
                references=refs,
            )  # predictions and preferences are expected to be a nested list of labels, not label_ids

        eval_metric = compute_metrics()
        logger.info(f"[VALID] epoch {epoch}: {eval_metric}")

        if eval_metric['f1'] > best_f1:     
            best_f1 = eval_metric['f1']
            best_epoch = epoch
            save_model(args, model)

    model, tokenizer = initialize_model(args, entity_embeddings)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "ckpt.pt"), map_location="cpu")["state_dict"])
    model.to(args.device)

    model.eval()
    for step, batch in enumerate(test_dataloader):
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        preds, refs = get_labels(predictions_gathered, labels_gathered)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    test_metric = compute_metrics()
    logger.info(f"[TEST] best epoch {best_epoch}: {test_metric}")

    with open(os.path.join(args.output_dir, "results.txt"), 'a+') as f:
        f.write("{}: {}\n".format(
            args.seed, test_metric
        ))


if __name__ == "__main__":
    main()