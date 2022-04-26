import json
from tqdm import tqdm
import numpy as np
import string
import collections

class InputFeatures(object):
    """
    A single set of features of data.
    """
    def __init__(self, qid, input_ids, attention_mask, token_type_ids, ent_pos, ent_mask, label_mask, wikidata_ids):
        self.qid = qid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.ent_pos = ent_pos
        self.ent_mask = ent_mask
        self.label_mask = label_mask
        self.wikidata_ids = wikidata_ids

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def norm_mask(input_mask):
    output_mask = np.zeros(input_mask.shape)
    for i in range(len(input_mask)):
        if not np.all(input_mask[i] == 0):
            output_mask[i] = input_mask[i] / sum(input_mask[i])
    return output_mask

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

def read_dataset(args):
    with open(args.train_file, 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)
    return input_data["data"]

def find_data(dataset, qid):
    for data in dataset:
        for paragraph in data["paragraphs"]:
            for qa in paragraph["qas"]:
                if qid == qa["id"]:
                    return qa

def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    max_ent_cnt=42,
    pad_token=0,
    doc_stride=128,
):
    def process_entities(text, tokens, entities, input_ids, wikidata_ids, ent_pos, ent_mask, ent_idx, doc_span_start=0):
        for entity in entities:
            if ent_idx >= max_ent_cnt:
                break

            start_idx, end_idx = char_to_token_idx(text, tokens, entity, len(input_ids))
            if start_idx is None or end_idx is None:
                continue
            start_idx -= doc_span_start
            end_idx -= doc_span_start

            if start_idx >= max_length or end_idx >= max_length or start_idx < len(input_ids) or end_idx < len(input_ids):
                continue

            # Sanity Check
            span = (input_ids + tokens.input_ids[doc_span_start:])[start_idx:end_idx+1]
            pred = tokenizer.decode(span)
            ground_truth = tokenizer.decode(tokenizer.encode(entity["text"], add_special_tokens=False))
            if ground_truth != pred:
                # print(f"Entity Mismatch GT:{ground_truth} vs Found:{pred}")
                continue

            wikidata_id = entity["id"]
            wikidata_ids.append(wikidata_id)

            for tok_idx in range(start_idx, end_idx+1):
                ent_pos[ent_idx][tok_idx] = 1
            ent_mask[ent_idx] = 1
            ent_idx += 1
        
        return wikidata_ids, ent_pos, ent_mask, ent_idx

    features = []

    for (ex_index, example) in enumerate(tqdm(examples, desc="Featurize...")):
        for paragraph in example["paragraphs"]:
            context_text = paragraph["context"]
            context_entities = paragraph["entity"]
            
            for qa in paragraph["qas"]:
                qid = qa["id"]
                question_text = qa["question"]
                question_entities = qa["question_entity"]

                query_tokens = tokenizer(question_text, add_special_tokens=False)
                context_tokens = tokenizer(context_text, add_special_tokens=False)
                all_doc_tokens = context_tokens.input_ids

                # The -2 accounts for [CLS], [SEP]
                max_tokens_for_doc = max_length - len(query_tokens.input_ids) - 2

                _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
                doc_spans = []
                start_offset = 0
                while start_offset < len(all_doc_tokens):
                    length = len(all_doc_tokens) - start_offset
                    if length > max_tokens_for_doc:
                        length = max_tokens_for_doc
                    doc_spans.append(_DocSpan(start=start_offset, length=length))
                    if start_offset + length == len(all_doc_tokens):
                        break
                    start_offset += min(length, doc_stride)

                for doc_span in doc_spans:
                    input_ids = []
                    token_type_ids = []
                    attention_mask = []

                    ent_pos = np.zeros((max_ent_cnt, max_length), dtype='int32') # Where is the mention of entity?
                    ent_mask = [0] * max_ent_cnt # Which entity is real? (not padding?)
                    wikidata_ids = []

                    ent_idx = 0 # Total number of entity in this QA data

                    # Add [CLS] token
                    input_ids.append(tokenizer.cls_token_id)
                    token_type_ids.append(0)
                    attention_mask.append(1)

                    wikidata_ids, ent_pos, ent_mask, ent_idx = process_entities(question_text, query_tokens, 
                                        question_entities, input_ids, wikidata_ids, ent_pos, ent_mask, ent_idx)
                    input_ids += query_tokens.input_ids
                    token_type_ids += query_tokens.token_type_ids
                    attention_mask += query_tokens.attention_mask

                    # No SEPerator for RE
                    
                    wikidata_ids, ent_pos, ent_mask, ent_idx = process_entities(context_text, context_tokens,
                                        context_entities, input_ids, wikidata_ids, ent_pos, ent_mask, ent_idx,
                                        doc_span.start)
                    
                    input_ids += context_tokens.input_ids[doc_span.start:doc_span.start+doc_span.length]
                    token_type_ids += context_tokens.token_type_ids[doc_span.start:doc_span.start+doc_span.length]
                    attention_mask += context_tokens.attention_mask[doc_span.start:doc_span.start+doc_span.length]

                    # TRUNCATE
                    if len(input_ids) >= max_length:
                        input_ids = input_ids[:max_length-1]
                        attention_mask = attention_mask[:max_length-1]
                        token_type_ids = token_type_ids[:max_length-1]

                    # [SEP]
                    input_ids += [tokenizer.sep_token_id]
                    attention_mask += [1]
                    token_type_ids += [0]

                    ent_pos = norm_mask(ent_pos)
                    ent_len = len(wikidata_ids)

                    label_mask = np.zeros((max_ent_cnt, max_ent_cnt), dtype='bool')
                    label_mask[:ent_len, :ent_len] = 1
                    for ent in range(ent_len):
                        label_mask[ent][ent] = 0
                    for ent in range(ent_len):
                        if np.all(ent_mask[ent] == 0):
                            label_mask[ent, :] = 0
                            label_mask[:, ent] = 0

                    # Padding
                    padding = [0] * (max_length - len(input_ids))
                    input_ids += padding
                    attention_mask += padding
                    token_type_ids += padding

                    features.append(
                        InputFeatures(
                            qid=qid,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            ent_pos=ent_pos,
                            ent_mask=ent_mask,
                            label_mask=label_mask,
                            wikidata_ids=wikidata_ids,
                        )
                    )
    return features