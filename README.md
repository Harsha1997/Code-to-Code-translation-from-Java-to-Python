## Code-to-Code-translation-from-Java-to-Python

This is the official PyTorch implementation for the following paper which is attached to the GitHub project: 
 **Code-to-Code translation from Java to Python using CodeBERT and CodeT5 models**

## Introduction

In this project, we look at two state-of-art models: CodeBERT and CodeT5 model which are capable of performing a plethora of software engineering tasks. We propose a novel technique to implement code translation from Java to Python using these pre-trained models. These models are not natively configured for such a code translation, thereby we engineer these models for the task at hand. We evaluate the results of these machine learning models to see how well they perform and discuss the reasons behind their performance and go over an in-depth analysis of the working (including pre-training) of these models and compare them with Facebook’s Transconder model. 

## Table of Contents: 
1) CodeT5
2) CodeBERT
3) Transcoder
4) Data
5) Dependencies
6) Folder Contents
7) Run Models

## CodeT5

CodeT5 builds on an encoder-decoder framework with the same architecture as T5. It aims to achieve generic representations for both programming language (PL) and natural language (NL) by utilizing pre-training on unlabeled source code. Please look into the paper titled [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/pdf/2109.00859.pdf) and the official [github repository](https://github.com/salesforce/CodeT5/) for more information on the workings of the CodeT5 model. 


## CodeBERT

CodeBERT is a pre-trained model for the programming language, which is a multi-programming-lingual model pre-trained on NL-PL pairs in 6 programming languages (Python, Java, JavaScript, PHP, Ruby, Go). CodeBERT captures the semantic relationship between natural language and programming language and generates general-purpose representations that can be used to support NL-PL comprehension and generation tasks.

CodeBERT follows BERT and RoBERTa and uses a multi-layer bidirectional Transformer as the model architecture of CodeBERT. For more information about the inner workings of the CodeBERT model please look into the paper titled [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155.pdf) and the official [github repository](https://github.com/microsoft/CodeBERT).

## Transcoder

TransCoder employs a sequence-to-sequence (seq2seq) model with attention, which consists of a transformer-based encoder and decoder. For all programming languages, it employs a single shared model. It is trained to utilize the initialization, language modeling, and back-translation methods of unsupervised machine translation. More details can be found in the paper titled [Unsupervised Translation of Programming Languages](https://arxiv.org/pdf/2006.03511.pdf) and the official [github repository](https://github.com/facebookresearch/CodeGen/blob/main/docs/transcoder.md#pre-trained-models).

## Data 

We have sourced the data from [AVATAR](http://web.cs.ucla.edu/~kwchang/bibliography/ahmad2021avatar/) dataset which contains the data scraped from multiple competitive coding platforms, where solutions to a set of algorithmic questions are solved by numerous users in
multiple languages (Java and Python will be explicitly used for our project). The platforms are as follows:
1) GeeksforGeeks
2) Leetcode
3) Codeforces
4) Atcoder
5) Google CodeJam
6) Project Euler

We also performed the necessary preprocessing steps to make the data easily ingestible by our deep learning models.

## Dependencies 

- pip install torch 
- pip install transformers
- Pytorch 1.7.1
- tensorboard 2.4.1
- transformers 4.6.1
- tree-sitter 0.2.2

Download the [Pre-trained checkpoints](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/pretrained_models) where we have various models of CodeT5 stored. You can use gsutils of Google Cloud as instructed below: 

```
# pip install gsutil
cd your-cloned-codet5-path

gsutil -m cp -r "gs://sfr-codet5-data-research/pretrained_models" .
```

Create a folder named "pretrained_models" inside the CodeT5 folder and then place the above-downloaded models in it. 

## Folder Contents

1. **CodeT5**
 - tokenizer: The tokenizer we apply to the data
 - data:  This folder has our training, testing, and validation data
 - evaluator: This contains our BLEU evaluation method
 - sh: This contains our results and the run_exp.py file that we need to run for our model. (PS: We removed the saved_models folder which contained our model checkpoints totaling 8.6 GB - which was too big for Github. So need to run the model again to get the test results)
 - .py files: The config, utils, and other files required to run
 
2. **CodeBERT** 
 - code: This contains the run.py, model.py, bleu.py, saved models, and output which is the main portion of our model (PS: Here also we removed the model, so need to train again for the results)
 - data: This folder has our training, testing, and validation data
 - evaluator: This contains our BLEU evaluation method
 - command.txt: This has the command we need to run
 
 ## Run Models 
 
 1. **CodeT5**
 
 Go to the sh folder and run the following command:
 ```
 python run_exp.py --model_tag codet5_base --task translate --sub_task java-python
 ```

(We are using codet5_base as our model, translation task, and the performing for java-python. The parameters are as of now hardcoded in the python files)

2. **CodeBERT**

Go to the Code-Code/code-to-code-trans/code and run the following command: 
 ```
#Training 
 
python run.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --train_filename ../data/train-java.java,../data/train-python.python --dev_filename ../data/valid-java.java,../data/valid-python.python --output_dir Train_Results --max_source_length 512 --max_target_length 512 --beam_size 5 --train_batch_size 16 --eval_batch_size 16 --learning_rate 5e-5 --train_steps 100000 --eval_steps 5000

#Testing

python run.py --do_test --model_type roberta --model_name_or_path roberta-base --config_name roberta-base  --tokenizer_name roberta-base  --load_model_path Train_Results/checkpoint-best-bleu/pytorch_model.bin --dev_filename ../data/valid-java.java,../data/valid-python.python --test_filename ../data/test-java.java,../data/test-python.python --output_dir Test_Results --max_source_length 512 --max_target_length 512 --beam_size 5 --eval_batch_size 16 
```
 
