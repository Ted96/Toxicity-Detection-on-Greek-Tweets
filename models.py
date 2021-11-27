##https://github.com/Ted96/Toxicity-Detection-on-Greek-Tweets
# bert transformers 
from typing import Tuple

import focal_loss
import tensorflow.python.keras.layers
from kerastuner import HyperModel
from tensorflow.python.ops.linalg_ops import self_adjoint_eig
from tensorflow_addons.metrics import F1Score
import numpy as np
import tensorflow.keras.backend as K
from keras.layers import Input
from sklearn.metrics import recall_score, precision_score
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.optimizers import Adam
# from tensorflow_addons.optimizers import AdamW
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dropout, Embedding, SpatialDropout1D, Bidirectional, GRU, GlobalMaxPooling1D, concatenate, GlobalAveragePooling1D,\
	Dense, BatchNormalization

# f1 loss https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

# https://www.kaggle.com/gtskyler/toxic-comments-bert/notebook
from transformers import TFBertModel, TFAutoModelForSequenceClassification, TFAutoModel, AdamWeightDecay, logging


logging.set_verbosity_error()

metric_auc = AUC(curve='ROC', name='AUC')
metric_aupr = AUC(curve='PR', name='AUPR')
metric_f1 = F1Score(num_classes=1, average='macro', name='f1')
metric_acc = BinaryAccuracy(name='acc')

def model_bert_focalloss(transformer, class_weights, LR=1e-5, name: str = None) -> Model:
	# input = ID + attention_mask

	transformer.compile(
		optimizer=Adam(learning_rate=LR),
		loss=focal_loss.SparseCategoricalFocalLoss(gamma=2., class_weight=class_weights, from_logits=True),
		metrics=[metric_acc]
	)
	transformer._name = 'Bert_sqCLF_focal_LR%.0e' % LR if name is None else name
	return transformer


def bert_4SequenceCLFN(transformer, LR=1e-4, name: str = None) -> Model:
	transformer.compile(
		optimizer=Adam(learning_rate=LR),
		loss=BinaryCrossentropy(from_logits=False),  # true
		metrics=[metric_acc]
	)
	transformer._name = 'Bert_4sqncCLFN_LR%.0e' % LR if name is None else name
	return transformer


def bert_model_MLP(transformer, LR=1e-5, max_length=64, name: str = None) -> Model:
	# input = ID + attention_mask
	_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
	_attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')

	__in = {'input_ids':_ids, 'attention_mask':_attention_mask}
	x = transformer(__in)[0]  # or transformer.bert
	cls_token = x[:, 0, :]
	x = Dropout(rate=0.25)(cls_token)
	x = Dense(256, activation='relu', name='Dense_0')(x)
	x = BatchNormalization(name='BN_0')(x)
	x = Dropout(rate=0.35)(x)
	x = Dense(32, activation='relu', name='Dense_1')(x)
	x = BatchNormalization(name='BN_1')(x)
	x = Dropout(rate=0.25)(x)
	__out = Dense(1, activation='sigmoid', name='output')(x)

	model = Model(inputs=__in, outputs=__out)
	model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=LR),
				  metrics=[metric_acc])
	model._name = 'Bert_CLS_mlp_LR%.0e' % LR if name is None else name

	return model


def model_bert_globalavgpool(transformer, max_length=64) -> Model:
	# classes = ??
	__num_classes = 1  ##########################################################

	# input = ID + attention_mask
	__in = Input(shape=(max_length,), name='input_ids', dtype='int32')

	x = transformer.bert(__in)[0]

	global_pool = GlobalAveragePooling1D()(x)

	__out = Dense(__num_classes, activation='sigmoid', name='output')(global_pool)

	model = Model(inputs=__in, outputs=__out)
	model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-5),
				  metrics=[metric_acc])
	return model


def bert_model_simple(transformer, LR=1e-5) -> Model:
	try:
		transformer.layers[2].activation = tensorflow.nn.sigmoid
	except IndexError:
		transformer.layers[1]._layers[2].activation = tensorflow.nn.sigmoid
	
	transformer = transformer
	transformer.compile(optimizer=Adam(learning_rate=LR),
				  loss=BinaryCrossentropy(from_logits=False),  # true
				  metrics=[metric_aupr],
				#run_eagerly=True
				  )
	transformer._name = 'Bert_simple_LR%.0e' % LR
	return transformer


# https://www.kaggle.com/akshat0007/bert-using-tensorflow-jigsaw-toxic-comment-data

def model_gru(maxlen, embedding_matrix: np.ndarray, LR=5e-4, trainable_embeddings=True, name: str = None):
	K.clear_session()

	__in = Input(shape=(maxlen,))
	x = Embedding(embedding_matrix.shape[0],
				  embedding_matrix.shape[1],
				  weights=[embedding_matrix],
				  input_length=maxlen,
				  trainable=trainable_embeddings)(__in)
	x = SpatialDropout1D(0.3)(x)
	x = Bidirectional(GRU(80, return_sequences=True))(x)
	x = Bidirectional(GRU(80, return_sequences=True))(x)
	avg_pool = GlobalAveragePooling1D()(x)
	max_pool = GlobalMaxPooling1D()(x)
	x = concatenate([avg_pool, max_pool])
	x = Dense(256, activation='relu', name='Dense_1')(x)
	x = Dropout(rate=0.2)(x)

	__out = Dense(1, activation='sigmoid')(x)

	model = Model(inputs=__in, outputs=__out)
	model.compile(optimizer=Adam(lr=LR),
				  loss='binary_crossentropy',
				  metrics=[metric_acc])
	model._name = 'GRU_LR%.0e' % LR if name is None else name

	return model


def clear_vram(model=None, transformer=None):
	import gc
	from tensorflow.keras import backend as K

	if model:
		del model
	if transformer:
		del transformer
	K.clear_session()
	gc.collect()

def load_model( _bert:str ,len_tokenizer, path_weights :str=None,LR =1e-5 , trainable=True)->Model:
	
	bert = TFAutoModelForSequenceClassification.from_pretrained(_bert, num_labels=1)
	bert.resize_token_embeddings(len_tokenizer)
	model = bert_model_simple(bert, LR=LR)
	if path_weights is not None:
		model.load_weights(filepath=path_weights)
	model.optimizer.lr.assign(LR)
	return model
