import json
import logging
import os
import argparse
from collections import defaultdict, Counter
import random
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)
# logging.disable(logging.WARNING)

from trainer import Trainer
from options import setup_args
from squad_metrics import (SquadResult, compute_predictions_logits,
                           squad_evaluate)

from utils import (
    QAProcessor,
    convert_examples_to_features_mp,
)

WEIGHTS_NAME = "pytorch_model.bin"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def initialize_model(args, entity_embeddings):
    print("Initialize model...")
    from models.KALA.modeling_bert import BertForExtractiveQA
    lm_ckpt = "bert-base-uncased"
    model_cls = BertForExtractiveQA
    tokenizer = AutoTokenizer.from_pretrained(lm_ckpt)
    args.tokenizer = tokenizer
    if args.checkpoint is None:
        model = model_cls.from_pretrained(lm_ckpt, 
                                        args=args, 
                                        entity_embeddings=entity_embeddings)
    else:
        print(f"Load checkpoint from {args.checkpoint}")
        config = AutoConfig.from_pretrained(lm_ckpt)
        model = model_cls.from_pretrained(args.checkpoint, 
                                        config=config,
                                        args=args, 
                                        entity_embeddings=entity_embeddings)

    return model, tokenizer

def run(args):
    set_seed(args.seed)
    args.device = 'cuda'

    entity_embeddings, wikidata_to_memory_map = load_entity_embeddings_memory(args)
    model, tokenizer = initialize_model(args, entity_embeddings)

    model.to(args.device)

    train_dataloader, _, _, _ = load_examples(args, "train")

    num_train_steps_per_epoch = len(train_dataloader)
    num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

    best_dev_score = [-1]
    best_weights = [None]
    results = {}

    def step_callback(model, global_step):
        if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
            epoch = int(global_step / num_train_steps_per_epoch - 1)
            dev_results = evaluate(args, model, fold="dev")
            tqdm.write("dev: " + str(dev_results))
            results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})
            if dev_results["f1"] > best_dev_score[0]:
                if hasattr(model, "module"):
                    best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
                else:
                    best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                best_dev_score[0] = dev_results["f1"]
                results["best_epoch"] = epoch
            model.train()

    if not args.do_eval:
        trainer = Trainer(
            args,
            model=model,
            dataloader=train_dataloader,
            num_train_steps=num_train_steps,
            step_callback=step_callback,
        )
        trainer.train()
        print(results)

        logger.info("Saving the model checkpoint to %s", args.output_dir)
        torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))

        # Load the best model on validation set for evaluation
        model, tokenizer = initialize_model(args, entity_embeddings)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)

    # Evaluate
    output_file = os.path.join(args.output_dir, "predictions.json")
    results = evaluate(args, model, fold="test", output_file=output_file)
    
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)
    
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    print(results)
    print(args.output_dir)
    return results

def evaluate(args, model, fold="dev", output_file=None):
    dataloader, examples, features, processor = load_examples(args, fold)
    tokenizer = args.tokenizer

    all_results = []
    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits, end_logits = outputs[0], outputs[1]
            outputs = (start_logits, end_logits)

        feature_indices = batch["feature_indices"]
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results

def load_entity_embeddings_memory(args):
    memory_path = os.path.join(args.data_dir, "train_entity_embeddings.pkl")

    with open(memory_path, 'rb') as f:
        entity_embeddings_memory = pickle.load(f)
    wikidata_to_memory_map = dict()
    entity_embeddings = []

    for key, value in entity_embeddings_memory.items():
        wikidata_to_memory_map[key] = len(entity_embeddings) + 1
        entity_embeddings.append(value)

    entity_embeddings = torch.from_numpy(np.stack(entity_embeddings, axis=0))
    print(f"# Entity Embeddings: {len(entity_embeddings)}")
    args.entity_embed_size = entity_embeddings.shape[-1]
    # args.entity_embed_size = 768
    args.wikidata_to_memory_map = wikidata_to_memory_map

    entity_embeddings = torch.cat([torch.zeros(1, entity_embeddings.shape[-1]), entity_embeddings], dim=0)
    return entity_embeddings, wikidata_to_memory_map

def load_examples(args, fold):
    wikidata_to_memory_map = args.wikidata_to_memory_map
    processor = QAProcessor(args)
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    pickle_name = "train_features_bert.pkl"
    pickle_path = os.path.join(args.pickle_folder, pickle_name)

    if not os.path.exists(pickle_path) or (fold == "dev" or fold == "test") or args.read_data:
        print("Creating features from the dataset...")
        features = convert_examples_to_features_mp(
            examples,
            args.tokenizer,
            args.max_seq_length,
            args.doc_stride,
            args.max_query_length,
            is_training=fold=="train"
        )
        if fold == "train":
            with open(pickle_path, 'wb+') as f:
                pickle.dump(features, f)
    else:
        print("Loading cached features...")
        with open(pickle_path, 'rb') as f:
            features = pickle.load(f)

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value):
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
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
        for batch_idx, (_, item) in enumerate(batch):
            nodes = [retrieve(node) for node in item.wikidata_ids]
            edge_index = [[len(graph_batch) + edge[0], len(graph_batch) + edge[1]] for edge in item.edge_index]
            # Reverse (Bidirectional)
            rev_edge_index = []
            rev_edge_attr = []
            for edge, edge_attr in zip(item.edge_index, item.edge_attr):
                rev_edge = [len(graph_batch) + edge[1], len(graph_batch) + edge[0]]
                if rev_edge in edge_index:
                    continue
                rev_edge_index.append(rev_edge)
                rev_edge_attr.append(edge_attr)
            
            graph_batch += [batch_idx] * len(nodes)
            batch_nodes += nodes
            batch_edge_index += edge_index
            batch_edge_attr += item.edge_attr
            
            batch_edge_index += rev_edge_index
            batch_edge_attr += rev_edge_attr
            batch_local_indicator += item.local_indicator

        ret = dict(
            input_ids=create_padded_sequence("input_ids", args.tokenizer.pad_token_id),
            attention_mask=create_padded_sequence("input_mask", 0),
            token_type_ids=create_padded_sequence("segment_ids", 0),
            mention_positions=create_padded_sequence("ent_pos", -1),
            nodes=torch.tensor(batch_nodes, dtype=torch.long),
            edge_index=torch.tensor(batch_edge_index, dtype=torch.long).t().reshape(2, -1),
            edge_attr=torch.tensor(batch_edge_attr, dtype=torch.long),
            graph_batch=torch.tensor(graph_batch, dtype=torch.long),
            local_indicator=torch.tensor(batch_local_indicator, dtype=torch.long)
        )
        if fold == "train":
            ret["start_positions"] = torch.stack([torch.tensor(getattr(o[1], "start_position"), dtype=torch.long) for o in batch])
            ret["end_positions"] = torch.stack([torch.tensor(getattr(o[1], "end_position"), dtype=torch.long) for o in batch])
        else:
            ret["feature_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)
        return ret

    if fold == "train":
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(
            list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, processor

if __name__ == "__main__":
    args = setup_args()
    run(args)
