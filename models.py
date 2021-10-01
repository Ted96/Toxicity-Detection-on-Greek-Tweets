# bert transformers
from typing import Tuple

from tensorflow_addons.metrics import F1Score
import numpy as np
import tensorflow.keras.backend as K
from keras.layers import Input
from sklearn.metrics import f1_score, recall_score, precision_score
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import AUC,BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dropout, Embedding, SpatialDropout1D, Bidirectional, GRU, GlobalMaxPooling1D, concatenate, GlobalAveragePooling1D,\
	Dense, BatchNormalization


#f1 loss https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

# https://www.kaggle.com/gtskyler/toxic-comments-bert/notebook
metric_auc = AUC(curve='ROC', name='AUC')
metric_aupr = AUC(curve='PR', name='AUPR')
metric_f1 = F1Score(num_classes=1,average='macro',name='F1')
metric_acc = BinaryAccuracy(name='acc')

class my_metrics(Callback):

	def __init__(self,validation_data:Tuple):
		super().__init__()
		self.x_val = validation_data[0]
		self.y_val = validation_data[1]
		self.val_precisions = []
		#self.val_recalls = []
		#self.val_f1s = []
	
	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict(self.x_val))).round()

		_val_f1 = round(f1_score(self.y_val, val_predict, average='macro'),3)
		self.val_f1s.append(_val_f1)

		#_val_recall = round(recall_score(val_targ, val_predict),3)
		#self.val_recalls.append(_val_recall)
		
		#_val_precision = round(precision_score(val_targ, val_predict),3)
		#self.val_precisions.append(_val_precision)
		
#		print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
		print(" — val_f1: %f " % _val_f1)

		return
	

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

	recall = true_positives / (all_positives + K.epsilon())
	return recall

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1(y_true, y_pred):
	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	return 2 * ((p * r) / (p + r + K.epsilon()))


def bert_4SequenceCLFN(transformer, LR=1e-4, name:str=None):
	
	transformer.compile(
		optimizer=Adam(learning_rate=LR),
		loss=SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)
	transformer._name =  'Bert_4SequenceCLFN_pure' if name is None else name
	return transformer

def bert_model1(transformer,  LR=1e-5, max_length=64, name:str = None) -> Model:

	# input = ID + attention_mask
	_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
	_attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
	
	__in = {'input_ids':_ids, 'attention_mask':_attention_mask}
	
	x = transformer.bert(__in)[0]
	cls_token = x[:,0,:]
	x = Dropout(rate=0.3)(cls_token)
	x = Dense(256, activation='relu', name='Dense_0')(x)
	x = BatchNormalization(name='BN_0')(x)
	x = Dropout(rate=0.3)(x)
	
	__out = Dense(1, activation='sigmoid', name='output')(x)

	model = Model(inputs=__in, outputs=__out)
	model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=LR),
				  metrics=[metric_acc, f1])
	model._name =  'Bert_mlp256_LR%.0e'%LR if name is None else name

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
				  metrics=[metric_acc, f1])
	return model


def bert_model2(transformer, max_length=64) -> Model:
	_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')

	x = transformer(_ids)[0]
	x = x[:, 0, :] # cls token
	
	__out = Dense(1, activation='sigmoid', name='output')(x)

	model = Model(inputs=_ids, outputs=__out)
	model.compile(Adam(lr=1e-5),
				  loss='binary_crossentropy',
				  metrics=[metric_acc, f1])

	return model


# https://www.kaggle.com/akshat0007/bert-using-tensorflow-jigsaw-toxic-comment-data

def bert_model3(transformer, max_length=64) -> Model:
	_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')

	x = transformer(_ids)[0][:, 0, :]  # cls token
	
	x = Dense(256, activation='relu' , name='Dense_0')(x)
	x = BatchNormalization(name='BN_0')(x)
	x= Dropout(rate=0.3)(x)
	
	x = Dense(64, activation='relu', name='Dense_1')(x)
	x = BatchNormalization(name='BN_1')(x)
	x = Dropout(rate=0.3)(x)

	__out = Dense(1, activation='sigmoid', name='output')(x)

	model = Model(inputs=_ids, outputs=__out)
	model.compile(Adam(lr=1e-4),
				  loss='binary_crossentropy',
				  metrics=[metric_acc, f1])

	return model


def model_gru(maxlen, vocab_size, embedding_matrix:np.ndarray , trainable_embeddings=True):
	K.clear_session()
	
	__in = Input(shape=(maxlen,))
	x = Embedding(vocab_size, 
				  embedding_matrix.shape[1],
				  weights=[embedding_matrix],
				  input_length=maxlen,
				  trainable=trainable_embeddings)(__in)
	x = SpatialDropout1D(0.2)(x)
	x = Bidirectional(GRU(80, return_sequences=True))(x)
	x = Bidirectional(GRU(80, return_sequences=True))(x)
	avg_pool = GlobalAveragePooling1D()(x)
	max_pool = GlobalMaxPooling1D()(x)
	x = concatenate([avg_pool, max_pool])
	x = Dense(128, activation='relu', name='Dense_1')(x)
	x = Dropout(rate=0.2)(x)
	
	__out = Dense(1, activation='sigmoid')(x)

	model = Model(inputs=__in, outputs=__out)
	model.compile(optimizer=Adam(lr=5e-4), 
				  loss='binary_crossentropy',
				  metrics=[metric_acc,f1])
	
	return model
