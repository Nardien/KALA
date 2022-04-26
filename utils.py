import json
import logging
import multiprocessing
import os
import shutil
from argparse import Namespace
import collections
import re
import string

from contextlib import closing
from multiprocessing.pool import Pool

from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer, whitespace_tokenize

from tqdm import tqdm

with open("./label_map.json") as f:
    label_map = json.load(f)

class SquadExample(object):
    """
       A single training/test example for the Squad dataset.
       For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 answers=[],
                 is_impossible=None,
                 context_id=None,
                 entities=None,
                 question_entities=None,
                 triplets=None,):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.answers = answers
        self.is_impossible = is_impossible
        self.context_id = context_id
        self.entities = entities,
        self.question_entities = question_entities
        self.triplets = triplets

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", doc_tokens: [%s]" % " ".join(self.doc_tokens)
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.end_position:
            s += ", end_position: %d" % self.end_position
        if self.is_impossible:
            s += ", is_impossible: %r" % self.is_impossible
        return s

class QAProcessor(object):
    def __init__(self, args):
        self.train_file = "train.json"
        self.dev_file = "dev.json"
        self.test_file = "test.json"

    def get_train_examples(self, data_dir):
        print(f"DataProcessor: {self.train_file}")
        input_file = os.path.join(data_dir, self.train_file)
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, is_training=True)

    def get_dev_examples(self, data_dir):
        print(f"DataProcessor: {self.dev_file}")
        input_file = os.path.join(data_dir, self.dev_file)
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, is_training=False)

    def get_test_examples(self, data_dir):
        print(f"DataProcessor: {self.test_file}")
        input_file = os.path.join(data_dir, self.test_file)
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, is_training=False)

    def _create_examples(self, input_data, is_training, version_2_with_negative=False):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        examples = []

        context_id = 0
        for entry in tqdm(input_data, total=len(input_data)):
            paragraphs = entry["paragraphs"]
            for paragraph in paragraphs:
                paragraph_text = paragraph["context"]
                entities = paragraph["entity"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    question_entities = qa["question_entity"]
                    if "triplets" not in qa.keys():
                        triplets = []
                    else:
                        triplets = qa["triplets"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if is_training:
                        if version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        answers = []
                        if not is_impossible:
                            answer = qa["answers"][0]
                            
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset +
                                                            answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(
                                doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""
                    else:
                        answers = qa["answers"]
                    
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        paragraph_text=paragraph_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        answers=answers,
                        is_impossible=is_impossible,
                        context_id=context_id,
                        entities=entities,
                        question_entities=question_entities,
                        triplets=triplets)
                    examples.append(example)
                context_id += 1
        return examples

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def char_to_token_idx(text, tokens, entity, length):
    start_char = entity["start"]
    end_char = entity["end"] - 1

    if start_char < 0 or end_char < 0:
        return None, None

    while end_char >= len(text) or is_whitespace(text[end_char]):
        end_char -= 1

    start_idx = tokens.char_to_token(start_char)
    end_idx = tokens.char_to_token(end_char)

    if start_idx is not None:
        start_idx += length
    if end_idx is not None:
        end_idx += length
    return start_idx, end_idx

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 qas_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 answer_text,
                 input_mask,
                 segment_ids,
                 context_id,
                 wikidata_ids,
                 ent_pos,
                 edge_index,
                 edge_attr,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 local_indicator=None,):
        self.unique_id = unique_id
        self.qas_id = qas_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.answer_text = answer_text
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.context_id = context_id
        ####
        self.wikidata_ids = wikidata_ids
        self.ent_pos = ent_pos
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.local_indicator = local_indicator

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_examples_to_features_mp(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training=True,
):
    pool_size = min(12, multiprocessing.cpu_count())
    chunk_size = 30
    worker_params = Namespace(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training
    )
    features = []
    unique_id = 1000000000
    with closing(Pool(pool_size, initializer=_initialize_worker, initargs=(worker_params,))) as pool:
        with tqdm(total=len(examples), desc="Featurize..") as pbar:
            for ret in pool.imap(_process_example, enumerate(examples), chunksize=chunk_size):
                for feature in ret:
                    feature.unique_id = unique_id
                    features.append(feature)
                    unique_id += 1
                pbar.update()
    return features

def _initialize_worker(_params):
    global params
    params = _params

def _process_example(args):
    example_index, example = args

    tokenizer = params.tokenizer
    max_seq_length = params.max_seq_length
    doc_stride = params.doc_stride
    max_query_length = params.max_query_length
    is_training = params.is_training

    def process_entities(text, tokens, entities, input_ids, wikidata_ids, ent_pos):
        for entity in entities:
            start_idx, end_idx = char_to_token_idx(text, tokens, entity, len(input_ids))
            if start_idx is None or end_idx is None:
                continue
            if start_idx >= len(ent_pos) or end_idx >= len(ent_pos):
                continue

            # Sanity Check
            span = (input_ids + tokens.input_ids)[start_idx:end_idx+1]
            pred = tokenizer.decode(span)
            ground_truth = tokenizer.decode(tokenizer.encode(entity["text"], add_special_tokens=False))
            if ground_truth != pred:
                # print(f"Question-Entity Mismatch GT:{ground_truth} vs Found:{pred}")
                continue

            wikidata_id = entity["id"]

            if wikidata_id in wikidata_ids:
                node_idx = wikidata_ids.index(wikidata_id)
            else:
                node_idx = len(wikidata_ids)
                wikidata_ids.append(wikidata_id)

            for tok_idx in range(start_idx, end_idx+1):
                ent_pos[tok_idx] = node_idx
        
        return wikidata_ids, ent_pos
    features = []
    
    context_id = example.context_id
    qas_id = example.qas_id
    query_tokens = tokenizer(example.question_text, add_special_tokens=False)
    question_entities = example.question_entities

    if len(query_tokens.input_ids) > max_query_length:
        query_tokens.input_ids = query_tokens.input_ids[0:max_query_length]
        query_tokens.attention_mask = query_tokens.attention_mask[0:max_query_length]

    context_tokens = tokenizer(example.paragraph_text, add_special_tokens=False)
    context_entities = example.entities
    if type(context_entities) == tuple: context_entities = context_entities[0]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    all_doc_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)
        
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens.input_ids) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
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

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        input_ids = []
        token_to_orig_map = {}
        token_is_max_context = {}
        token_type_ids = [] # segment_ids

        ent_pos = [-1] * (len(query_tokens.input_ids) + 1) # Where is the mention of entity (all-in-one)
        wikidata_ids = [] # Wikidata IDs of each entity (for Node information)
        ent_idx = 0 # Total number of entity in this QA data

        # Add [CLS] token
        input_ids.append(tokenizer.cls_token_id)
        token_type_ids.append(0)

        wikidata_ids, ent_pos = process_entities(example.question_text, query_tokens, question_entities,
                                                    input_ids, wikidata_ids, ent_pos)

        input_ids += query_tokens.input_ids
        for _ in range(len(query_tokens.input_ids)):
            token_type_ids += [0]

        # Add [SEP] token
        input_ids += [tokenizer.sep_token_id]
        token_type_ids += [0]
        ent_pos = ent_pos + [-1] # [SEP]

        # Merge question and context
        c_ent_pos = [-1] * max(max_seq_length, len(context_tokens.input_ids))
        wikidata_ids, c_ent_pos = process_entities(example.paragraph_text, context_tokens, context_entities,
                                                    [], wikidata_ids, c_ent_pos)

        # Doc Span
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(
                input_ids)] = tok_to_orig_index[split_token_index]
            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                    split_token_index)
            token_is_max_context[len(input_ids)] = is_max_context
            input_ids.append(all_doc_ids[split_token_index])
            ent_pos.append(c_ent_pos[split_token_index])
            token_type_ids.append(1)

        # TRUNCATE
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length-1]
            token_type_ids = token_type_ids[:max_seq_length-1]
        ent_pos = ent_pos[:len(input_ids)]

        # Last [SEP]
        input_ids += [tokenizer.sep_token_id]
        token_type_ids += [1]
        ent_pos += [-1]

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
        wikidata_ids = local_wikidata_ids
        ent_pos = local_ent_pos

        attention_mask = [1] * len(input_ids)

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)
            ent_pos.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(ent_pos) == max_seq_length

        start_position = None
        end_position = None

        if is_training and not example.is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens.input_ids) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if is_training and example.is_impossible:
            start_position = 0
            end_position = 0

        edge_index = []
        edge_attr = []
        local_indicator = [1 for _ in range(len(wikidata_ids))]
        if len(example.triplets) > 0:
            for triplet in example.triplets:
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

        features.append(
            InputFeatures(
                unique_id=None,
                qas_id=qas_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=attention_mask, # attention_mask
                answer_text=example.orig_answer_text,
                segment_ids=token_type_ids, # token_type_ids
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible,
                context_id=context_id,
                wikidata_ids=wikidata_ids,
                ent_pos=ent_pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                local_indicator=local_indicator,
            )
        )
    return features
