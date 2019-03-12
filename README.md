# For setup pytorch-BERT please follow huggingface-README.md. once you setup your environment , clone this repo and follow below execution steps.

This Repo is having code for downward NLP applications like Named Entity Recognition(NER),Intent classification using multi label classification,...etc.

this repo uses code from <code>https://github.com/huggingface/pytorch-pretrained-BERT<code>

# Future work : sentiment analysis, QA retrival, text entailment, text similarity..etc.

# to use existing trained model execute below code :
```
python examples/nerTest_usingBert.py --data_dir data/conll2003/ --bert_model bert-base-uncased --do_lower_case --do_train --do_eval --do_test --do_pred --task_name NER --output_dir custom_models/ner_output_conll2003

python examples/nerTest_usingBert.py --data_dir data/ncbi-disease/conll/ --bert_model bert-base-uncased --do_lower_case --do_train --do_eval --do_test --do_pred --task_name NER --output_dir custom_models/ner_output_disease

python examples/intentTest_usingBert.py --data_dir data/toxic_cmt/ --bert_model bert-base-uncased --do_lower_case --do_train --do_eval --do_pred --task_name intent_multilabel --output_dir custom_models/intent_output_toxic
```
# For trying out your own..(you can change any input parameters as you like) :
```
python examples/nerTest_usingBert.py --data_dir data/conll2003/ --bert_model bert-base-uncased --do_lower_case --do_train --do_eval --do_test --do_pred --task_name NER --output_dir custom_models/model-directory-name

python examples/nerTest_usingBert.py --data_dir data/ncbi-disease/conll/ --bert_model bert-base-uncased --do_lower_case --do_train --do_eval --do_test --do_pred --task_name NER --output_dir custom_models/model-directory-name

python examples/intentTest_usingBert.py --data_dir data/toxic_cmt/ --bert_model bert-base-uncased --do_lower_case --do_pred --task_name intent_multilabel --output_dir custom_models/model-directory-name
```
