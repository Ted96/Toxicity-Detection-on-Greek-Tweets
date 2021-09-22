# %% imports
import re
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.python.data import Dataset
from tqdm.notebook import tqdm


class dataset_loader:

	def __init__(self, dataset: str):
		self.__which_dataset: str = dataset
		if dataset == 'ogtd':
			self.df = pd.read_csv('datasets/ogtd/train.tsv', delimiter='\t', names=['id', 'text', 'label'], header=0)
			self.df_labels = pd.read_csv('datasets/ogtd/test_labels.csv', names=['id', 'label'])
			self.df_test = pd.read_csv('datasets/ogtd/test.tsv', delimiter='\t', names=['id', 'text'], header=0)

			self.df_test = self.df_test.merge(self.df_labels, on='id')

			self.df_test['label'] = self.df_test['label'].apply(lambda value:1 if value == 'OFF' else 0 if value == 'NOT' else -1)
			self.df['label'] = 		self.df['label']	 .apply(lambda value:1 if value == 'OFF' else 0 if value == 'NOT' else -1)
			del self.df_labels

		if dataset == 'jigsaw18':
			self.df = pd.read_csv('datasets/jigsaw18/train.csv')
			self.df = self.df.sample(frac=0.2).rename(columns={'comment_text':'text'})

		self.x_train, self.x_test = self.df['text'].to_numpy(), self.df_test['text'].to_numpy()
		self.y_train, self.y_test = self.df['label'].to_numpy(), self.df_test['label'].to_numpy()
		self.x_val, self.y_val = [], []

		self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)))

	def reload(self, new_dataset: str = None):
		del self.df, self.df_test, self.x_test, self.y_test, self.x_val, self.y_val, self.x_train, self.y_train
		self.__init__(new_dataset if new_dataset else self.__which_dataset)
		return self

	def clean_data(self, remove_urls=True, remove_hastags=True, remove_accents=True):

		def clean_text(txt: str):
			if remove_hastags:
				txt = re.sub('#\S*', '', txt)
			if remove_urls:
				txt = re.sub('http\S+', '', txt)
			if remove_accents:
				txt = remove_greek_accents(txt)

			return txt

		for _list in self.x_train, self.x_test:
			for i, txt in enumerate(_list):
				_list[i] = clean_text(txt)  # todo list = _map_(func , list)

		# self.df['text'] = self.df['text'].apply(lambda x:clean_text(x))
		# self.df_test['text'] = self.df_test['text'].apply(lambda x:clean_text(x))
		return self

	def fast_encode(self, tokenizer, max_length=128, chunk_size=256):
		# doesnt affect dataframes

		tokenizer.enable_truncation(max_length=max_length)
		tokenizer.enable_padding(length=max_length)

		def tokenizer_encode_batch(texts) -> np.ndarray:
			all_ids = []

			for i in tqdm(range(0, len(texts), chunk_size)):
				text_chunk = texts[i:i + chunk_size]
				encs = tokenizer.encode_batch(text_chunk)
				all_ids.extend([enc.ids for enc in encs])

			return np.array(all_ids)

		self.x_train = tokenizer_encode_batch(self.x_train)
		self.x_test = tokenizer_encode_batch(self.x_test)
		if self.x_val:
			self.x_val = tokenizer_encode_batch(self.x_val)

		return self


	def get_XY_lists(self, split_val=0.0) -> (List, List, List, List, List, List):
		# doesnt affect dataframes
		# x_train, x_test = self.df['text'].tolist(), self.df_test['text'].tolist()
		# y_train, y_test = self.df['label'].tolist(), self.df_test['label'].tolist()

		if split_val == 0:
			return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test
		else:
			self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=split_val, shuffle=True,
																				  random_state=42)

			return np.array(self.x_train), np.array(self.y_train), self.x_val, np.array(self.y_val), self.x_test, self.y_test

	def get_XY_TFDatasets(self, batch_size, split_val=0.0):

		print('Creating TF Datasets', '(w/out validation)' if split_val == 0 else '(w/ validation)')
		if split_val == 0:
			dataset_train = (
				Dataset
					.from_tensor_slices((self.x_train, self.y_train))
					.shuffle(2048)
					.batch(batch_size)
					.prefetch(-1)
			)

		else:
			self.x_train, self.x_val, self.y_train, self.y_val =\
				train_test_split(self.x_train, self.y_train, test_size=split_val, shuffle=True, random_state=42)

			dataset_train = (
				Dataset
					.from_tensor_slices((self.x_train, self.y_train))
					.shuffle(2048)
					.batch(batch_size)
					.prefetch(-1)
			)
			dataset_val = (
				Dataset
					.from_tensor_slices((self.x_val, self.y_val))
					.batch(batch_size)
					.cache()
					.prefetch(-1)
			)

		dataset_test = (
			Dataset
				.from_tensor_slices((self.x_test, self.y_test))
				.batch(batch_size)
				.prefetch(-1)
		)

		return dataset_train, dataset_val if split_val != 0 else None, dataset_test


def remove_greek_accents(txt: str):
	txt = txt.replace('ά', 'α')
	txt = txt.replace('έ', 'ε')
	txt = txt.replace('ή', 'η')
	txt = txt.replace('ό', 'ο')
	txt = txt.replace('ώ', 'ω')
	txt = txt.replace('ί', 'ι')
	txt = txt.replace('ΐ', 'ι')
	txt = txt.replace('ϊ', 'ι')
	txt = txt.replace('ϋ', 'υ')
	txt = txt.replace('ΰ', 'υ')
	txt = txt.replace('ύ', 'υ')
	return txt


def fast_encode(texts: List[str], tokenizer, max_length=128, chunk_size=256) -> np.ndarray:
	tokenizer.enable_truncation(max_length=max_length)
	tokenizer.enable_padding(length=max_length)

	all_ids = []

	for i in tqdm(range(0, len(texts), chunk_size)):
		text_chunk = texts[i:i + chunk_size]
		encs = tokenizer.encode_batch(text_chunk)
		all_ids.extend([enc.ids for enc in encs])

	return np.array(all_ids)


def load_embeddings(word_index, embedding_file: str):
	embeddings_index = {}

	with open(embedding_file, encoding='UTF-8') as f:
		for line in f:
			values = line.split()
			word = values[0]
			if word in word_index:
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs

	total_not_found = 0
	emb_dim = coefs.shape[0]
	embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		if embedding_vector is None:
			print(word)
			total_not_found += 1

	# if word not in embeddings_index and remove_accents(word) in embeddings_index:
	# 	print('-->', word)
	# #todo set UNK to mean? set to 0 ?

	print('#Word Vectors: %s' % len(embeddings_index),
		  '  |  #Not matched: %s' % total_not_found,
		  '  |  embd Matrix:', embedding_matrix.shape)

	return embedding_matrix


# def get_padded_texts_and_embeddings_matrix(x_train,x_val, x_test, embedding_file:str , maxlen = 48):
# 	tokenizer = Tokenizer(oov_token='__UNK__')
# 	tokenizer.fit_on_texts(list(x_train))
# 
# 	x_train = tokenizer.texts_to_sequences(x_train)
# 	#new_x_val = tokenizer.texts_to_sequences(x_val)
# 	x_test = tokenizer.texts_to_sequences(x_test)
# 	
# 	# Pad the sentences
# 	x_train = pad_sequences(x_train, maxlen=maxlen)
# 	#x_val = pad_sequences(x_train, maxlen=maxlen)
# 	x_test = pad_sequences(x_test, maxlen=maxlen)
# 
# 	word_index = tokenizer.word_index
# 	vocab_size = len(word_index) + 1
# 
# 	embedding_matrix = load_embeddings(word_index, embedding_file)
# 	
# 	return x_train , x_test,embedding_matrix
