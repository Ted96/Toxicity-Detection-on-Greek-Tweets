# %% imports | read dataset | load bert
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizer, TFAutoModel, TFBertModel, TFBertForSequenceClassification, AutoTokenizer, TFRobertaForSequenceClassification

from data_processing import dataset_loader, load_embeddings, fast_encode
from models import bert_model1, bert_model2, bert_model3, model_gru, bert_4SequenceCLFN
from utils import evaluate_model, find_best_classification_threshold, displot_tokenized_sentences


tf.random.set_seed(3407)
np.random.seed(3407)

MAX_LEN_SEQUENCE = 64
MAX_LEN_EMBEDDINGS = 40
BATCH_SIZE = 64
# config = BertConfig.from_pretrained("bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", config=config)
# bert = TFAutoModel.from_pretrained("bert-base-uncased")
# bert_greek.bert.trainable=False #todo
bert_greek: TFBertModel
tokenizer = None

# try:
# 	bert_greek = TFBertModel.from_pretrained('./bert_pretrained')
# 	tokenizer = BertWordPieceTokenizer('bert_pretrained/vocab.txt', lowercase=True)
# 	print('Local models: Bert + Tokenizer Loaded!')
# except (OSError, FileNotFoundError):
# 	print('Downloading Bert + Tokenizer...')
# 	tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
# 	tokenizer.save_pretrained('./bert_pretrained')
# 	bert_greek = TFAutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
# 	bert_greek.save_pretrained('./bert_pretrained')

# read datasets
data = dataset_loader('ogtd')
# %% tokenizer
MAX_LEN_SEQUENCE = 160
_bert :str = ["nlpaueb/bert-base-greek-uncased-v1" ,  			#0 greek
			  "gealexandri/palobert-base-greek-uncased-v1" ,	#1 palo
			  "bert-base-multilingual-uncased",					#2 mbert
			  "roberta-base"][0]								#3 xlmr
# tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
tokenizer = AutoTokenizer.from_pretrained(_bert)
#tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

x_train, y_train, x_val, y_val, x_test, y_test = data.reload().clean_data(to_lower=True).get_XY_lists(split_val=0.08)

ttr = tokenizer(x_train.tolist(), max_length=MAX_LEN_SEQUENCE, truncation=True, padding=True, return_tensors='tf', return_token_type_ids=True)
ttv = tokenizer(x_val.tolist(), max_length=MAX_LEN_SEQUENCE, truncation=True, padding=True, return_tensors='tf', return_token_type_ids=True)
tte = tokenizer(x_test.tolist(), max_length=MAX_LEN_SEQUENCE, truncation=True, padding=True, return_tensors='tf', return_token_type_ids=True)
displot_tokenized_sentences(ttr['input_ids'], pad_token=tokenizer.vocab[tokenizer.pad_token])  #pad:  1=xlmr   0=bert   

# %% M #1
bert_greek = TFBertModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = bert_model1(bert_greek, LR=6e-6 , max_length=MAX_LEN_SEQUENCE)  #LR 2e-4 maxlen64 +CW-> f0.839

hs = model.fit(x={'input_ids':ttr['input_ids'], 'attention_mask':ttr['attention_mask']}, y=y_train,
			   validation_data=({'input_ids':ttv['input_ids'], 'attention_mask':ttv['attention_mask']}, y_val),
			   epochs=7,
			   class_weight=data.class_weights,
			   callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True, mode='min', verbose=1)],
			   batch_size=BATCH_SIZE,
			   workers=6)
evaluate_model(model,
			   {'input_ids':ttv['input_ids'], 'attention_mask':ttv['attention_mask']}, y_val,
			   {'input_ids':tte['input_ids'], 'attention_mask':tte['attention_mask']}, y_test)

# %% M #2
BATCH_SIZE = 64
# bert_greek.bert.trainable = False  # todo
dataset_train, dataset_val, dataset_test = data.clean_data(remove_accents=False).fast_encode(tokenizer, max_length=MAX_LEN_SEQUENCE).get_XY_TFDatasets(
	batch_size=BATCH_SIZE, split_val=0.08)
model = bert_model2(bert_greek, MAX_LEN_SEQUENCE)
hs = model.fit(
	dataset_train,
	validation_data=dataset_val,
	# batch_size=BATCH_SIZE,
	steps_per_epoch=dataset_train.cardinality().numpy(),  # x_train.shape[0] // BATCH_SIZE,
	epochs=8,
	class_weight=data.class_weights,
	callbacks=[EarlyStopping(monitor='val_f1', patience=2, restore_best_weights=True, mode='max')],
	workers=6
)

# %% M #3
model = bert_model3(bert_greek, MAX_LEN_SEQUENCE)
# bert_greek.bert.trainable = False  # todo

hs = model.fit(
	dataset_train,
	validation_data=dataset_val,
	batch_size=BATCH_SIZE,
	steps_per_epoch=dataset_train.cardinality().numpy(),
	epochs=7,
	# class_weight=data.class_weights,
	callbacks=[EarlyStopping(monitor='val_f1', patience=5, restore_best_weights=True, mode='max')],
	workers=6
)

# %% fasttext embeddings
embedding_file = 'datasets/fasttext_greek_300.vec'

x_train, y_train, x_val, y_val, x_test, y_test =\
	data.reload().clean_data(remove_accents=False).get_XY_lists(0.08)

tokenizer = Tokenizer(oov_token='__UNK__')
tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

# Pad the sentences
x_train = pad_sequences(x_train, maxlen=MAX_LEN_EMBEDDINGS)
x_val = pad_sequences(x_val, maxlen=MAX_LEN_EMBEDDINGS)
x_test = pad_sequences(x_test, maxlen=MAX_LEN_EMBEDDINGS)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

embedding_matrix = load_embeddings(word_index, embedding_file)

# %% M gru 

BATCH_SIZE = 512
model = model_gru(MAX_LEN_EMBEDDINGS, vocab_size, embedding_matrix, trainable_embeddings=True)
hs = model.fit(
	x=x_train,
	y=y_train,
	validation_data=(x_val, y_val),
	batch_size=BATCH_SIZE,
	steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
	epochs=30,
	# class_weight=data.class_weights,
	callbacks=[EarlyStopping(monitor='val_f1', patience=10, restore_best_weights=True, mode='max', verbose=1)],
	workers=4
)

# %% M sklearn 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


x_train, y_train, x_val, y_val, x_test, y_test = data.reload().clean_data(remove_accents=True).get_XY_lists()

# %% M sklearn  (SVM / RF / LIGHTGBM  +tfidf )
vectorizer = TfidfVectorizer(max_features=1500)  # 1500

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)
classifiers = [LGBMClassifier(n_estimators=120, max_depth=4), RandomForestClassifier(n_estimators=39), LinearSVC(C=0.1)]
for clf in classifiers:
	clf.fit(x_train_tfidf, y_train)
	y_pred = clf.predict(x_test_tfidf)
	print(clf.__class__.__name__, ' f1 =', round(f1_score(y_test, y_pred, average='macro'), 3), '\n')

# %% M bert4sequence 
BATCH_SIZE = 8
# bert_greek = TFRobertaForSequenceClassification.from_pretrained("gealexandri/palobert-base-greek-uncased-v1",num_labels=2)
#bert_greek = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
bert_greek = TFRobertaForSequenceClassification.from_pretrained(_bert)

tf.config.run_functions_eagerly(True)

model = bert_4SequenceCLFN(bert_greek, LR=1e-6)
hs = model.fit(x={'input_ids':ttr['input_ids'], 'attention_mask':ttr['attention_mask']}, y=y_train,
			   validation_data=({'input_ids':ttv['input_ids'], 'attention_mask':ttv['attention_mask']}, y_val),
			   epochs=2,
			   class_weight=data.class_weights,
			   callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True, mode='min', verbose=1)],
			   batch_size=BATCH_SIZE,
			   workers=6)

evaluate_model(model,
			   {'input_ids':ttv['input_ids'], 'attention_mask':ttv['attention_mask']}, y_val,
			   {'input_ids':tte['input_ids'], 'attention_mask':tte['attention_mask']}, y_test)


# %%

dataset = tte
if dataset == ttv:
	y_true = y_val
elif dataset == tte:
	y_true = y_test

y_pred_proba = model.predict(x={'input_ids':dataset['input_ids'], 'attention_mask':dataset['attention_mask']}, verbose=1, workers=6)
logits = y_pred_proba.logits
# %%
y_pred = [logit[1] - logit[0] for logit in logits]

threshold = find_best_classification_threshold(y_true, y_pred)
y_pred = [0 if logit[1] - logit[0] < 0 else 1 for logit in logits]

print('f1 =', round(f1_score(y_true, y_pred, average='macro'), 3))

# %% M bert4sequence XLMR


# %%

unique, counts = np.unique(ttr['input_ids'], return_counts=True)
d = dict(zip(unique, counts))
print('#Unknown:', d[1])

text = 'Είναι ένας <mask> άνθρωπος.'

encodings = tokenizer.encode(text)
# [101, 357, 449, 103, 964, 121, 102]

print(encodings, '=\n', tokenizer.convert_ids_to_tokens(encodings))

# %%

import sys
import gc


def actualsize(input_obj):
	memory_size = 0
	ids = set()
	objects = [input_obj]
	while objects:
		new = []
		for obj in objects:
			if id(obj) not in ids:
				ids.add(id(obj))
				memory_size += sys.getsizeof(obj)
				new.append(obj)
		objects = gc.get_referents(*new)
	return memory_size

print( actualsize(x_test) / 1000000 , 'MB')