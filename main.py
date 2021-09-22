# %% imports

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
# bert transformers
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizer, TFAutoModel, TFBertModel , TFBertForSequenceClassification

from data_processing import dataset_loader, load_embeddings
from models import bert_model2, bert_model3 , model_gru

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(3407)

MAX_LEN_SEQUENCE = 64
MAX_LEN_EMBEDDINGS = 40
BATCH_SIZE = 64
# config = BertConfig.from_pretrained("bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", config=config)
# bert = TFAutoModel.from_pretrained("bert-base-uncased")
# bert_greek.bert.trainable=False #todo
bert_greek: TFBertModel
tokenizer = None

try:
	bert_greek = TFBertModel.from_pretrained('./bert_pretrained')
	tokenizer = BertWordPieceTokenizer('bert_pretrained/vocab.txt', lowercase=True)
	print('Local models: Bert + Tokenizer Loaded!')
except (OSError, FileNotFoundError):
	print('Downloading Bert + Tokenizer...')
	tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
	tokenizer.save_pretrained('./bert_pretrained')
	bert_greek = TFAutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
	bert_greek.save_pretrained('./bert_pretrained')

# %% read datasets
data = dataset_loader('ogtd')
# %% model #2  //Batch128  maxlen64   // valf1: 0.73   testf1:

BATCH_SIZE=64
#bert_greek.bert.trainable = False  # todo
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

# %% model #3      //Batch128  maxlen64    !!batchsize=1024 runs!
model = bert_model3(bert_greek, MAX_LEN_SEQUENCE)
#bert_greek.bert.trainable = False  # todo

hs = model.fit(
	dataset_train,
	validation_data=dataset_val,
	batch_size=BATCH_SIZE,
	steps_per_epoch=dataset_train.cardinality().numpy(),
	epochs=7,
	#class_weight=data.class_weights,
	callbacks=[EarlyStopping(monitor='val_f1', patience=5, restore_best_weights=True, mode='max')],
	workers=6
)

#%%  GRU fasttext embeddings
embedding_file = 'datasets/fasttext_greek_300.vec'

x_train, y_train, x_val, y_val, x_test, y_test = \
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

#%%  model gru 

BATCH_SIZE = 512
model = model_gru(MAX_LEN_EMBEDDINGS ,vocab_size,embedding_matrix,trainable_embeddings=True) 
hs = model.fit(
	x=x_train,
	y=y_train,
	validation_data=(x_val,y_val),
	batch_size=BATCH_SIZE,
	steps_per_epoch= x_train.shape[0]//BATCH_SIZE,
	epochs=30,
	#class_weight=data.class_weights,
	callbacks=[EarlyStopping(monitor='val_f1', patience=10, restore_best_weights=True, mode='max',verbose=1)],
	workers=4
)

# %% predict test

y_pred_proba = model.predict(x_test, verbose=True)

#%%

y_pred = [0 if y < 0.5 else 1 for y in y_pred_proba]
f1_score(data.y_test, y_pred)


# %%
def find_best_cuttof(y_true, probability_predictions):
	start = 0.3
	end = 1
	step = 0.005
	current = start
	max_f1 = 0
	best_cuttof = 0.5
	while current <= end:
		y_pred_new = [0 if y < current else 1 for y in probability_predictions]
		_f1 = f1_score(y_true, y_pred_new)
		if _f1 > max_f1:
			max_f1 = _f1
			best_cuttof = current

		current += step
	print('max f1:', round(max_f1, 3), '  cuttof=', round(best_cuttof,3))


find_best_cuttof(data.y_test, np.array(y_pred_proba))


# %%

def test(x):
	puncts = [
		'*', '+', '\\', '•', '~', '£',
		'·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
		'█', '½', 'à', '…',
		'“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
		'¥', '▓', '—', '‹', '─',
		'▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
		'¾', 'Ã', '⋅', '‘', '∞',
		'∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
		'¹', '≤', '‡', '√', ]

	total = []
	for i in puncts:
		if i in x:
			total.append(i)

	return total


# preprocess todo: clean these:  ' " `  << >> - $
df['howmany'] = df['text'].apply(test)

# %% tokenizer fast

# x_train = fast_encode(x_train, tokenizer, max_length=MAX_LEN_SEQUENCE)
# x_val = fast_encode(x_val, tokenizer, max_length=MAX_LEN_SEQUENCE)
# x_test = fast_encode(x_test, tokenizer, max_length=MAX_LEN_SEQUENCE)

#%% dataset


# dataset_train = (
#     tf.data.Dataset
#     .from_tensor_slices((x_train, y_train))
#     .repeat()
#     .shuffle(2048)
#     .batch(BATCH_SIZE)
#     .prefetch(-1)
# )
# 
# dataset_val = (
#     tf.data.Dataset
#     .from_tensor_slices((x_val, y_val))
#     .batch(BATCH_SIZE)
#     .cache()
#     .prefetch(-1)
# )

#%% model #1

# BATCH_SIZE = 32
# model  = bert_model1(bert_greek)
# # OOM MAX BS=32 maxlen=128   |   BS=64 maxlen=64
# hs = model.fit(
#     x={'input_ids': input_tokenized['input_ids'], 'attention_mask': input_tokenized['attention_mask']},
#     y={'output': y_train},
#     #validation_split=0.2,
#     batch_size=BATCH_SIZE,
# 	epochs=1,
#     workers=6
# 	)
# 
# BATCH_SIZE=32
# model = model_bert_globalavgpool(bert_greek, MAX_LEN_SEQUENCE)
# 
# hs = model.fit(
#     dataset_train,
#     validation_data=dataset_val,
# 	batch_size=BATCH_SIZE,
# 	steps_per_epoch= x_train.shape[0] // BATCH_SIZE,
#     epochs=6,
# 	class_weight=class_weights,
# 	callbacks=[EarlyStopping(monitor='val_f1', patience=2, restore_best_weights=True, mode='max')],
# 	workers=6
# )

# %%  	

from matplotlib import pyplot as plt

def plot_history(hs: dict, metric='accuracy', title: str = ''):
	print()
	# sns.set()
	plt.style.use('dark_background')
	plt.rcParams['figure.figsize'] = [14, 8]
	plt.rcParams['font.size'] = 15
	plt.clf()

	colors = ['#52006A', '#CD113B', '#50DD93', '#FF7600', '#D62AD0']
	for label, color in zip(hs, colors):
		plt.plot(hs[label][metric], label='{0:s} train {1:s}'.format(label, metric), linewidth=1.5, color=color)
		plt.plot(hs[label]['val_{0:s}'.format(metric)], label='{0:s} val_ {1:s}'.format(label, metric), linewidth=2, linestyle='-.', color=color)

		if metric in ['accuracy', 'f1', 'AUC']:
			_max = np.max(hs[label]['val_{0:s}'.format(metric)])
			plt.plot(int(np.argmax(hs[label]['val_{0:s}'.format(metric)])), _max, 'x', markersize=10, color='#FAFF00')
		if metric in ['loss']:
			_min = np.min(hs[label]['val_{0:s}'.format(metric)])
			plt.plot(int(np.argmin(hs[label]['val_{0:s}'.format(metric)])), _min, 'x', markersize=10, color='#FAFF00')

	_epochs = max([len(hs[label]['loss']) for label in hs])
	x_ticks = np.arange(0, _epochs + 1, step=2)
	x_ticks[0] += 1
	plt.xticks(x_ticks)
	plt.title(title)

	plt.xlabel('Epochs')
	plt.ylabel(metric)
	plt.legend(loc=0)
	plt.show()

plot_history({'':hs.history}, metric='f1', title='gru+mlp trainableEmbd LRe-3')

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
vectorizer = TfidfVectorizer(max_features=1500)  #1500

x_train, y_train, x_val, y_val, x_test, y_test = data.reload().clean_data().get_XY_lists()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# %%
classifiers = [LGBMClassifier(n_estimators=120,max_depth=4), RandomForestClassifier(n_estimators=39), LinearSVC(C=0.1)]
for clf in classifiers:
	clf.fit(x_train,y_train)
	y_pred = clf.predict(x_test)
	print(clf.__class__.__name__,' f1 =',round(f1_score(y_test,y_pred),3),'\n')


# models to test
# https://arxiv.org/pdf/1912.06872.pdf		#Towards Robust Toxic Content Classification

# bert implementations																	#notes
# https://www.kaggle.com/nayansakhiya/text-classification-using-bert						simple
# https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-with-huggingface-and-keras			tf.dataset | fast encode | cls token
# https://www.kaggle.com/gtskyler/toxic-comments-bert/									average pooling
# https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras			distilbert | tf.dataset  | cls token

# https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671
# https://www.tensorflow.org/text/tutorials/fine_tune_bert
# https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22
# https://www.analyticsvidhya.com/blog/2020/10/simple-text-multi-classification-task-using-keras-bert/
# https://medium.com/analytics-vidhya/bert-in-keras-tensorflow-2-0-using-tfhub-huggingface-81c08c5f81d8

# embeddings
# https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b
# http://archive.aueb.gr:7000/resources/
# 
# 
# classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# # x_train,y_train , x_val,y_val,x_test , y_test = data.clean_data().get_XY_lists(0.08)
# # x , test , matrix = get_padded_texts_and_embeddings_matrix(x_train , None , x_test , '../datasets/fasttext_greek_300.vec')