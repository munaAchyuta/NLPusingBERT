import argparse
#import csv
import logging
import os
import random
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from seqeval.metrics import f1_score, accuracy_score, classification_report

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(contends) == 0:# and words[-1] == '.':
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")


    def get_labels(self):
        #return ["B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
        #return ['B-Disease', 'O', 'I-Disease','X','[CLS]','[SEP]']
        return ['TEST_ENTITY', 'ACTION', 'schedule_num', 'PROBLEM_ID', 'SUBJECT', 'state', 'action', 'SCHEDULE_NUM', 'subject', 'problem_id', 'O', 'form_num', 'problem_type', 'input_type','X','[CLS]','[SEP]']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[1]#tokenization.convert_to_unicode(line[1])
            label = line[0]#tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples


def convert_examples_to_features_test(example, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    if True:
        textlist = example['text_a'].split(' ')
        labellist = ['X']*len(example['text_a'].split(' '))
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        # tokens = tokenizer.tokenize(example.text)
        print(tokens)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        #label_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
            #label_mask.append(0)
        print("ntokens : ",ntokens)
        #print("len of input_mask : ",len(input_mask))
        #print("len of segment_ids : ",len(segment_ids))
        #print("len of label_ids : ",len(label_ids))
        #print("len of input_ids[0] : ",len(input_ids[0]))
        #print("len of input_mask[0] : ",len(input_mask[0]))
        #print("len of segment_ids[0] : ",len(segment_ids[0]))
        #print("len of label_ids[0] : ",len(label_ids[0]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features

def convert_examples_to_features_pred(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    #example = {'text_a':'','text_b':'','label':'','guid':'}
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        # tokens = tokenizer.tokenize(example.text)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        #label_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
            #label_mask.append(0)
        #print("len of input_ids : ",len(input_ids))
        #print("len of input_mask : ",len(input_mask))
        #print("len of segment_ids : ",len(segment_ids))
        #print("len of label_ids : ",len(label_ids))
        #print("len of input_ids[0] : ",len(input_ids[0]))
        #print("len of input_mask[0] : ",len(input_mask[0]))
        #print("len of segment_ids[0] : ",len(segment_ids[0]))
        #print("len of label_ids[0] : ",len(label_ids[0]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    print("label_map : ",label_map)
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        # tokens = tokenizer.tokenize(example.text)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        #label_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
            #label_mask.append(0)
        #print("len of input_ids : ",len(input_ids))
        #print("len of input_mask : ",len(input_mask))
        #print("len of segment_ids : ",len(segment_ids))
        #print("len of label_ids : ",len(label_ids))
        #print("len of input_ids[0] : ",len(input_ids[0]))
        #print("len of input_mask[0] : ",len(input_mask[0]))
        #print("len of segment_ids[0] : ",len(segment_ids[0]))
        #print("len of label_ids[0] : ",len(label_ids[0]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_accc(preds,labels):
    pred_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    #print(np.sum(pred_flat == labels_flat),"======", len(labels_flat))
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(model):
    # create model
    #model = Sequential()
    #model.add(Dense(12, input_dim=8, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
    # Compile model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.train()
    #for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    #tr_loss = 0
    #nb_tr_examples, nb_tr_steps = 0, 0
    #for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    #batch = tuple(t.to(device) for t in batch)
    #input_ids, input_mask, segment_ids, label_ids = batch
    X1=np.zeros((115,128)) 
    X2=np.zeros((115,128)) 
    X3=np.zeros((115,128))
    Y=np.zeros((115,128))
    loss = model(X1,X2,X3,Y)
    #loss = model()
    if n_gpu > 1:
        loss = loss.mean() # mean() to average on multi-gpu.
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    if args.fp16:
        optimizer.backward(loss)
    else:
        loss.backward()
                
    # added clip
    if args.clip is not None:
        _ = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

    tr_loss += loss.item()
    #nb_tr_examples += input_ids.size(0)
    #nb_tr_steps += 1
    #if (step + 1) % args.gradient_accumulation_steps == 0:
    #    if args.fp16:
    #        # modify learning rate with special warm up BERT uses
    #        # if args.fp16 is False, BertAdam is used that handles this automatically
    #        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
    #        for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr_this_step
    #    optimizer.step()
    #    optimizer.zero_grad()
    #    global_step += 1
    optimizer.step()
    optimizer.zero_grad()
    #global_step += 1
    return model
# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
# load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]

from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import TransformerMixin

class KerasInputFormatter(_BaseComposition, TransformerMixin):

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y = None):
        for n, t in self.transformers:
            t.fit(X, y)

        return self

    def transform(self, X):
        return { n: t.transform(X) for n, t in self.transformers }

    def get_params(self, deep=True):
        return self._get_params('transformers', deep=deep)

    def set_params(self, **kwargs):
        self._set_params('transformers', **kwargs)
        return self

from sklearn.base import BaseEstimator
from keras.models import Model
from keras.layers import Input

class KerasModel(BaseEstimator):
    def __init__(self, optimizer = 'sgd'):
        self.optimizer = optimizer # an example of a tunable hyperparam

    def fit(self, X, y):
        input_one = Input(name = 'input_one', shape = X['input_one'].shape[1:])
        input_two = Input(name = 'input_two', shape = X['input_two'].shape[1:])
        input_three = Input(name = 'input_three', shape = X['input_three'].shape[1:])

        output = y # define model here

        self.model = Model(inputs = [input_one, input_two, input_three], outputs = output)

        self.model.compile(self.optimizer, 'mse')
        self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='/home/adzuser/user_achyuta/BERT_NER_Test/BERT-NER/NERdata/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='NER',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='ner_output',
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--do_pred",
                        action='store_true',
                        help="Whether to run pred on the pred set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,#3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--clip',
                        type=float,
                        default=0.5,
                        help="gradient clipping")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    
    parser.add_argument('--text_a', type=str, default='', help="input text_a.")
    parser.add_argument('--text_b', type=str, default='', help="input text_b.")
    
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "ner": NerProcessor
    }

    num_labels_task = {
        "ner": 17#6#12
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_pred:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        #train_examples = train_examples[:1000]
        print("train_examples :: ",len(list(train_examples)))
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
    #imodel = BertForSequenceClassification.from_pretrained(args.bert_model,
    #          cache_dir=cache_dir,
    #          num_labels = num_labels)
    model = BertForTokenClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)

    if args.fp16:
        model.half()
    #model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)

        #all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        #all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        #all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        #all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
        all_input_ids = [f.input_ids for f in train_features]
        all_input_mask = [f.input_mask for f in train_features]
        all_segment_ids = [f.segment_ids for f in train_features]
        all_label_ids = [f.label_id for f in train_features]
        
        # convert to cuda
        #all_input_ids = all_input_ids.to(device)
        #all_input_mask = all_input_mask.to(device)
        #all_segment_ids = all_segment_ids.to(device)
        #all_label_ids = all_label_ids.to(device)
        
        
        #train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        #if args.local_rank == -1:
        #    train_sampler = RandomSampler(train_data)
        #else:
        #    train_sampler = DistributedSampler(train_data)
        #train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        # create model
        #model.train()
        model = KerasClassifier(build_fn=create_model(model), verbose=0)
        # define the grid search parameters
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        
        #keras_input = KerasInputFormatter([
        #    ('input_one', all_input_ids),
        #    ('input_two', all_segment_ids),
        #    ('input_three', all_input_mask),])
        #input_one = Input(name = 'input_one', shape = (128,))
        #input_two = Input(name = 'input_two', shape = (128,))
        #input_three = Input(name = 'input_three', shape = (128,))
        #input_four = Input(name = 'input_four', shape = (128,))

        #output = y # define model here

        #self.model = Model(inputs = [input_one, input_two, input_three], outputs = output)
        
        grid_result = grid.fit([[all_input_ids, all_segment_ids, all_input_mask]], [all_label_ids])
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
             print("%f (%f) with: %r" % (mean, stdev, param))

        
        '''
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                #print(input_ids.shape,input_mask.shape,segment_ids.shape,label_ids.shape)
                #print(input_ids[0])
                #print(label_ids[0])
                #logits = model(input_ids, segment_ids, input_mask)
                #import pdb;pdb.set_trace()
                #print(logits.view(-1, num_labels).shape, label_ids.view(-1).shape)
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                
                # added clip
                if args.clip is not None:
                    _ = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
       '''

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        #model = BertForSequenceClassification(config, num_labels=num_labels)
        model = BertForTokenClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        #model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
        # Load a trained model and config that you have fine-tuned
        print('for eval only......................')
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        config = BertConfig(output_config_file)
        #model = BertForSequenceClassification(config, num_labels=num_labels)
        model = BertForTokenClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        #import pdb;pdb.set_trace()
        print("dev_eaxmples :: ",len(list(eval_examples)))
        eval_features = convert_examples_to_features_pred(eval_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], [] 
        #predictions1 , true_labels1 = [], []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            # get index till '[SEP]'
            #print("label_list index SEP : ",label_list.index('[SEP]'))
            pred_xx = [list(p) for p in np.argmax(logits, axis=2)]
            pred_xx = [i[:i.index(label_list.index('[SEP]'))]for i in pred_xx]
            label_ids_xx = [i[:i.index(label_list.index('[SEP]'))]for i in label_ids.tolist()]
            #print(label_ids_xx)
            #print(pred_xx)

            # new add
            tmp_s = [max(len(i), len(j)) for i,j in zip(label_ids_xx,pred_xx)]
            tmp_u = [(i+[31]*(k-len(i)) if len(i) !=k else i,j+[31]*(k-len(j)) if len(j) !=k else j) for i,j,k in zip(label_ids_xx,pred_xx,tmp_s)]
            tmp_d1 = [h[0] for h in tmp_u]
            tmp_d2 = [h[1] for h in tmp_u]

            #print([list(p) for p in np.argmax(logits, axis=2)][:5])
            #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            tmp_eval_accuracy = flat_accc(pred_xx, label_ids_xx)
            #tmp_eval_accuracy = flat_accc(tmp_d1, tmp_d2)
            predictions.extend(tmp_d2)
            true_labels.append(tmp_d1)
            #predictions1.extend(pred_xx)
            #true_labels1.append(label_ids_xx)
            
            #print("tmp accuracy : ",tmp_eval_accuracy)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        loss = tr_loss/nb_tr_steps if args.do_train else None

        pred_tags = [[label_list[p_i] if p_i!=31 else 'XXX' for p_i in p] for p in predictions]
        valid_tags = [[label_list[l_ii] if l_ii!=31 else 'YYY' for l_ii in l_i] for l in true_labels for l_i in l ]
        print("valid_tags : ",valid_tags[:10])
        print("pred_tags : ",pred_tags[:10])
        print("Validation F1-Score: {}".format(f1_score(valid_tags, pred_tags)))
        print("Validation accuracy_score : {}".format(accuracy_score(valid_tags, pred_tags)))
        print("Validation classification_report : {}".format(classification_report(valid_tags, pred_tags)))
        
        #print("X Validation F1-Score: {}".format(f1_score(true_labels1, predictions1)))
        #print("X Validation accuracy_score : {}".format(accuracy_score(true_labels1, predictions1)))
        #print("X Validation classification_report : {}".format(classification_report(true_labels1, predictions1)))


        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}
        print(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        print('test examples len : {}'.format(len(eval_examples)))
        #import pdb;pdb.set_trace()
        eval_features = convert_examples_to_features_pred(eval_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], [] 

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            # get index till '[SEP]'
            #print("label_list index SEP : ",label_list.index('[SEP]'))
            pred_xx = [list(p) for p in np.argmax(logits, axis=2)]
            pred_xx = [i[:i.index(label_list.index('[SEP]'))]for i in pred_xx]
            label_ids_xx = [i[:i.index(label_list.index('[SEP]'))]for i in label_ids.tolist()]
            #print(label_ids_xx)
            #print(pred_xx)

            # new add
            tmp_s = [max(len(i), len(j)) for i,j in zip(label_ids_xx,pred_xx)]
            tmp_u = [(i+[31]*(k-len(i)) if len(i) !=k else i,j+[31]*(k-len(j)) if len(j) !=k else j) for i,j,k in zip(label_ids_xx,pred_xx,tmp_s)]
            tmp_d1 = [h[0] for h in tmp_u]
            tmp_d2 = [h[1] for h in tmp_u]

            #print([list(p) for p in np.argmax(logits, axis=2)][:5])
            #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            tmp_eval_accuracy = flat_accc(pred_xx, label_ids_xx)
            #tmp_eval_accuracy = flat_accc(tmp_d1, tmp_d2)
            predictions.extend(tmp_d2)
            true_labels.append(tmp_d1)
            #print("tmp accuracy : ",tmp_eval_accuracy)
            test_loss += tmp_eval_loss.mean().item()
            test_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        test_loss = test_loss / nb_eval_steps
        test_accuracy = test_accuracy / nb_eval_steps
        loss = tr_loss/nb_tr_steps if args.do_train else None

        pred_tags = [[label_list[p_i] if p_i!=31 else 'XXX' for p_i in p] for p in predictions]
        valid_tags = [[label_list[l_ii] if l_ii!=31 else 'YYY' for l_ii in l_i] for l in true_labels for l_i in l ]
        print("valid_tags : ",valid_tags[:10])
        print("pred_tags : ",pred_tags[:10])
        print("Test F1-Score: {}".format(f1_score(valid_tags, pred_tags)))
        print("Test accuracy_score : {}".format(accuracy_score(valid_tags, pred_tags)))
        print("Test classification_report : {}".format(classification_report(valid_tags, pred_tags)))
        
        #print("X Test F1-Score: {}".format(f1_score(true_labels, predictions)))
        #print("X Test accuracy_score : {}".format(accuracy_score(true_labels, predictions)))
        #print("X Test classification_report : {}".format(classification_report(true_labels, predictions)))


        result = {'test_loss': test_loss,
                  'test_accuracy': test_accuracy,
                  'global_step': global_step,
                  'loss': loss}
        print(result)
        output_test_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_pred and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        #eval_examples = processor.get_dev_examples(args.data_dir)
        model.eval()
        while True:
            print('enter a text to get NER. otherwise press Ctrl+C to close session.')
            text_a = input('>>>')
            #"Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday . ."
            eval_examples = {'text_a':text_a,'text_b':"The foodservice pie business does not fit our long-term growth strategy .",'label':'1','guid':'12345'}

            eval_features = convert_examples_to_features_test(eval_examples, label_list, args.max_seq_length, tokenizer)
            
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            #model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels = [], [] 

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                pred_xx = [list(p) for p in np.argmax(logits, axis=2)]
                pred_xx = [i[:i.index(label_list.index('[SEP]'))] for i in pred_xx]

                print(pred_xx)
                print([[label_list[p_i] if p_i!=31 else 'XXX' for p_i in p] for p in pred_xx]) 


if __name__=='__main__':
   main()
