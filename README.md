## A Novel Joint Extraction Framework for Knowledge base Question Answering

* Introduction
  >  In this paper, we combine the transformer encoder and convolutional neural networks to propose a Gated-Attention-based Joint Training Model (Ga-JTM) for relation and entity joint extraction.
* Data
  > 1. Glove [glove.6B.300d.txt] —— Download pre-trained word vectors from <https://nlp.stanford.edu/projects/glove/>
  > 2. All the pre-processed dataset and the trained model can download from https://pan.baidu.com/s/1TXFExkJvcknxsG9fdrQVyw code: 8i10

* Train/Test
  > 1. Run `python main.py  -train --data_type wq/sq` to train our model.
  > 2. Run `python main.py -test --data_type wq/sq` to test the joint training model.
