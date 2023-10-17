# KALA: Knowledge-Augmented Language Model Adaptation

Official Code Repository for the paper "KALA: Knowledge-Augmented Language Model Adaptation" (NAACL 2022): https://arxiv.org/abs/2204.10555.

In this repository, we implement the training code for our KALA framework as well as the preprocessing code of the task datasets.

## Abstract
<img align="middle" width="900" src="https://github.com/Nardien/KALA/blob/master/images/concept_fig.png">

Pre-trained language models (PLMs) have achieved remarkable success on various natural language understanding tasks.
Simple fine-tuning of PLMs, on the other hand, might be suboptimal for domain-specific tasks because they cannot possibly cover knowledge from all domains.
While adaptive pre-training of PLMs can help them obtain domain-specific knowledge, it requires a large training cost.
Moreover, adaptive pre-training can harm the PLM's performance on the downstream task by causing catastrophic forgetting of its general knowledge.
To overcome such limitations of adaptive pre-training for PLM adaptation, we propose a novel domain adaptation framework for PLMs coined as Knowledge-Augmented Language model Adaptation (KALA),
which modulates the intermediate hidden representations of PLMs with domain knowledge, consisting of entities and their relational facts.
We validate the performance of our KALA on question answering and named entity recognition tasks on multiple datasets across various domains.
The results show that, despite being computationally efficient, our KALA largely outperforms adaptive pre-training.

## Installation
Python version: 3.6.0 (thanks to @jiangxinke)
```bash
python -m pip install -r requirements.txt
```
If you face any issues on installing torch-geometric library, please refer to [torch-geometric github](https://github.com/pyg-team/pytorch_geometric).

## Dataset
In regard to QA tasks, please download the dataset from the below links, and place it into `dataset` folder. Regarding NER tasks, you can directly download the dataset in the knowledge graph construction step below with `./KGC/preprocess_ner.py`.

Due to the license issue, we cannot directly provide the emrQA dataset. Please download the emrQA from here: https://github.com/panushri25/emrQA.

Download path for the NewsQA dataset: [NewsQA](https://drive.google.com/file/d/1TZCOm6lGKaz4fm_QaCrZladN-7YJkjt2/view?usp=sharing)

## Knowledge Graph Construction
To construct the Knowledge Graph (KG) for each dataset, please follow the below instructions.
If you are interested in how the KG is constructed, please refer to codes in `KGC` folder.

1. Entity Extraction

Please download `en_core_web_md` in spacy library and spaCy-entity-linker library here: https://github.com/egerber/spaCy-entity-linker.

Be sure to run `python -m spacy_entity_linker "download_knowledge_base"` command to download the knowledge base for entity linker (thanks to @jiangxinke).

They are needed to correctly extract entities from the context, and to map them to the corresponding Wikidata ids.

Below are the example scripts for preprocessing the NewsQA dataset.

```bash
python preprocess.py NewsQA train
python preprocess.py NewsQA dev
python preprocess.py NewsQA test
```

2. Knowledge Graph Construction

Before start, please download the pre-trained Relation Extraction model and place it on the `RE_checkpoint` directory. 

Download path: [model.pth](https://drive.google.com/file/d/1XrUUb6aDWTTPAV_CfTBWGh6sYr89w9OW/view?usp=sharing)

Below are the example scripts for KG construction on the NewsQA dataset.

```bash
python construct.py --domain NewsQA --fold train
python construct.py --domain NewsQA --fold dev
python construct.py --domain NewsQA --fold test 
```

While the example scripts above are for question answering tasks, you can similarly run the code for named entity recognition tasks via the file: {filename}_ner.py.

Furthermore, you can directly import the preprocessed datasets for NER tasks by downloading the file [here](https://1drv.ms/u/s!Aj5JerV8SMDyi59jrrKrSK-HUw6wGQ?e=paPdRX), and then place it on the `./KGC/TASK` directory. For example, in regard to the CONLL2003 dataset, `./KGC/TASK/conll2003/results/{train_0.05.json, validation_0.05.json, test_0.05.json, train_entity_embeddings_0.05.pkl}`.

## Training
After preprocessing, the preprocessed dataset files are stored at `./KGC/TASK/$DATA_NAME` with a default setting.

Then, run the below code to train the model for the NewsQA dataset over our KALA framework.

```bash
python run_qa.py --data_dir ./KGC/TASK/NewsQA
```

This code runs 2 epochs of training on the NewsQA dataset. It takes approximately 3 hours on a single GeForce RTX 2080 Ti GPU.

## Fine-tuned Checkpoint

We provide the fine-tuned checkpoint of our KALA on the NewsQA dataset.

Download path: [NewsQA_ckpt.zip](https://drive.google.com/file/d/1yVXmAboH-8Es_7fNmwChFypdXrXAZ3Yl/view?usp=sharing)

After unzip the downloaded file, run the following code to reproduce the result on the NewsQA dataset.

```bash
python run_qa.py --do_eval --checkpoint ./NewsQA_ckpt/ --data_dir ./KGC/TASK/NewsQA
```

## Citation
If you found the provided code with our paper useful, we kindly request that you cite our work.
```BibTex
@inproceedings{kang2022kala,
  author       = {Minki Kang and
                  Jinheon Baek and
                  Sung Ju Hwang},
  editor       = {Marine Carpuat and
                  Marie{-}Catherine de Marneffe and
                  Iv{\'{a}}n Vladimir Meza Ru{\'{\i}}z},
  title        = {{KALA:} Knowledge-Augmented Language Model Adaptation},
  booktitle    = {Proceedings of the 2022 Conference of the North American Chapter of
                  the Association for Computational Linguistics: Human Language Technologies,
                  {NAACL} 2022, Seattle, WA, United States, July 10-15, 2022},
  pages        = {5144--5167},
  publisher    = {Association for Computational Linguistics},
  year         = {2022},
  url          = {https://doi.org/10.18653/v1/2022.naacl-main.379},
}
```
