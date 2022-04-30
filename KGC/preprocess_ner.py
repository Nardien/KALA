import sys
import os
import spacy
import en_core_web_md
from tqdm import tqdm
import json

from datasets import load_dataset

def extract_entity(nlp, text, entities):
    doc = nlp(text)
    all_linked_entities = doc._.linkedEntities

    if len(all_linked_entities) > 0:
        for entity in all_linked_entities:
            span = entity.get_span()
            start = span.start_char
            end = span.end_char
            entities.append({'text':span.text, 'start':start, 'end':end, 'id':entity.get_id()})

def preprocess(datadir, domain, fold):
    raw_datasets = load_dataset(domain, cache_dir=datadir)
    dataset = raw_datasets[fold]

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)

    num_entities = 0
    doc_id_to_entities = {}

    for entry in tqdm(dataset):

        entities = []

        tokens = entry['tokens']
        doc_text = " ".join(tokens)

        extract_entity(nlp, doc_text, entities)
        doc_id_to_entities[entry['id']] = entities

        num_entities += len(entities)

    assert len(dataset) == len(doc_id_to_entities)

    _filename_out = f"{fold}_entity.json"
    filename_out = os.path.join(datadir, domain, _filename_out)

    with open(filename_out, 'w') as outfile:
        json.dump(doc_id_to_entities, outfile)

    print(f"Save Done: {filename_out}")
    print(f"Num Avg Entities: {num_entities / len(dataset)}")

    with open(filename_out, 'r') as infile:
        loaded_doc_id_to_entities = json.load(infile)

    assert doc_id_to_entities == loaded_doc_id_to_entities

if __name__ == "__main__":
    datadir = '../dataset'
    domain = sys.argv[1]
    fold = sys.argv[2]

    assert domain in ['conll2003', 'wnut_17', 'ncbi_disease']
    assert fold in ['train', 'validation', 'test']

    preprocess(datadir, domain, fold)