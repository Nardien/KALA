# Code Guide for ACL 2022 Submission "KALA: Knowledge-Augmented Language Model Adaptation"

Since the Knowledge Graph Construction step needs external Entity Linker with cached index file and the BERT-base checkpoint fine-tuned on the Relation Extraction task, we do not expect that the reviewer to run the preprocessing step. Instead, we provide the preprocessed version of pre-constructed KG for NewsQA dataset including train, dev, and test set.

## Dataset
Download path:

## Knowledge Graph Construction
If you are interested in how the KG is constructed, pleases refer to codes in `KGC` folder.

1. Entity Extraction

Please download `en_core_web_md` in spacy library and spaCy-entity-linker library here: https://github.com/egerber/spaCy-entity-linker.

They are needed to correctly extract entities from the context and map them to the corresponding Wikidata ids.

Belows are the example scripts for preprocessing the NewsQA dataset.

```bash
python preprocess.py NewsQA train
python preprocess.py NewsQA dev
python preprocess.py NewsQA test
```

2. Knowledge Graph Construction
Before start, please download pre-trained Relation Extraction model and place it on the `RE_checkpoint` directory. 

Download path:

Belows are the example scripts for KG construction on the NewsQA dataset.

```bash
python construct.py --domain NewsQA --fold train
python construct.py --domain NewsQA --fold dev
python construct.py --domain NewsQA --fold test 
```

## Training
After preprocessing, run below code to train the model for NewsQA dataset, within KALA framework.

```bash
python run_qa.py
```

This code runs 2 epochs of training on NewsQA dataset. It costs approximately 3 hours on single GeForce RTX 2080 Ti GPU.
