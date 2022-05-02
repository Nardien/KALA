# KALA: Knowledge-Augmented Language Model Adaptation

Official Code Repository for the paper "KALA: Knowledge-Augmented Language Model Adaptation" (NAACL 2022): https://arxiv.org/abs/2204.10555.

In this repository, we implement the training code for our KALA framework inclduing the preprocessing the task dataset.

## Abstract
<img align="middle" width="900" src="https://github.com/Nardien/KALA/blob/master/images/concept_fig.png">

Pre-trained language models (PLMs) have achieved remarkable success on various natural language understanding tasks.
Simple fine-tuning of PLMs, on the other hand, might be suboptimal for domain-specific tasks because they cannot possibly cover knowledge from all domains.
While adaptive pre-training of PLMs can help them obtain domain-specific knowledge, it requires a large training cost.
Moreover, adaptive pre-training can harm the PLM's performance on the downstream task by causing catastrophic forgetting of its general knowledge.
To overcome such limitations of adaptive pre-training for PLM adaptation, we propose a novel domain adaptation framework for PLMs coined as Knowledge-Augmented Language model Adaptation (KALA),
which modulates the intermediate hidden representations of PLMs with domain knowledge, cosisting of entities and their relational facts.
We validate the performance of our KALA on question answering and named entity recognition tasks on multiple datasets across various domains.
The results show that, despite being computationally efficient, our KALA largely outperforms adaptive pre-training.

## Installation
```bash
python -m pip install -r requirements.txt
```
If you face any issues on installing torch-geometric library, please refer to [torch-geometric github](https://github.com/pyg-team/pytorch_geometric).

## Dataset
Please download the proper dataset from below, and place it into `dataset` folder.

Due to the license issue, we cannot directly provide the emrQA dataset. Please download the emrQA from here: https://github.com/panushri25/emrQA.

Download path: [NewsQA](https://drive.google.com/file/d/1TZCOm6lGKaz4fm_QaCrZladN-7YJkjt2/view?usp=sharing)

## Knowledge Graph Construction
To construct the KG for each dataset, pleases follow the below instructions.
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

Download path: [model.pth](https://drive.google.com/file/d/1XrUUb6aDWTTPAV_CfTBWGh6sYr89w9OW/view?usp=sharing)

Belows are the example scripts for KG construction on the NewsQA dataset.

```bash
python construct.py --domain NewsQA --fold train
python construct.py --domain NewsQA --fold dev
python construct.py --domain NewsQA --fold test 
```

While the example scripts above are for question answering tasks, you can similarly run the code for named entity recognition tasks via the file: {filename}_ner.py.

## Training
After preprocessing, the preprocessed dataset files are stored at `./KGC/TASK/$DATA_NAME` with a default setting.

Then, run below code to train the model for NewsQA dataset, within KALA framework.

```bash
python run_qa.py --data_dir ./KGC/TASK/NewsQA
```

This code runs 2 epochs of training on NewsQA dataset. It costs approximately 3 hours on single GeForce RTX 2080 Ti GPU.

## Fine-tuned Checkpoint

We provide the fine-tuned checkpoint of KALA on NewsQA dataset.

Download path: [NewsQA_ckpt.zip](https://drive.google.com/file/d/1yVXmAboH-8Es_7fNmwChFypdXrXAZ3Yl/view?usp=sharing)

After unzip the downloaded file, run following code to reproduce the result on NewsQA dataset.

```bash
python run_qa.py --do_eval --checkpoint ./NewsQA_ckpt/ --data_dir ./KGC/TASK/NewsQA
```

## Citation
If you found the provided code with our paper useful, we kindly requiest that you cite our work.
```BibTex
@article{kang2022kala,
  title={KALA: Knowledge-Augmented Language Model Adaptation},
  author={Kang, Minki and Baek, Jinheon and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2204.10555},
  year={2022}
}
```
