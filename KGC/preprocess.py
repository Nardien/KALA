# Preprocessing code for extracting entities from the QA dataset
import sys
import os
import spacy
import en_core_web_md
from tqdm import tqdm
import json

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
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)

    _filename = f"{fold}.json"

    filename = os.path.join(datadir, domain, _filename)

    print(f"Preprocessing {filename}")
    
    with open(filename, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    dataset = input_data["data"]

    for entry in tqdm(dataset):
        paragraphs = entry["paragraphs"]

        for paragraph in paragraphs:
            entities = []

            paragraph_text = paragraph["context"]
            extract_entity(nlp, paragraph_text, entities)
            paragraph["entity"] = entities
            
            for qa in paragraph["qas"]:
                question_entities = []
                question_text = qa["question"]
                extract_entity(nlp, question_text, question_entities)
                qa["question_entity"] = question_entities

    print(len(dataset))

    _filename_out = f"{fold}_entity.json"
    filename_out = os.path.join(datadir, domain, _filename_out)

    with open(filename_out, 'w') as outfile:
        json.dump(input_data, outfile)

    print(f"Save Done: {filename_out}")

if __name__ == "__main__":
    datadir = "../dataset"
    domain = sys.argv[1]
    fold = sys.argv[2]
    preprocess(datadir, domain, fold)
