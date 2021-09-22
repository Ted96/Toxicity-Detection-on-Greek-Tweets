# bert transformers
import numpy as np
import tensorflow.keras.backend as K
from keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout, Embedding, SpatialDropout1D, Bidirectional, GRU, GlobalMaxPooling1D, concatenate, GlobalAveragePooling1D,\
	Dense, BatchNormalization


#f1 loss https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric

# https://www.kaggle.com/gtskyler/toxic-comments-bert/notebook
metric_auc = AUC(curve='ROC', name='AUC')
metric_aupr = AUC(curve='PR', name='AUPR')


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


def bert_model1(transformer, max_length=64) -> Model:
	# classes = ??
	__num_classes = 1  ##########################################################

	# input = ID + attention_mask
	_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
	_attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
	__in = {'input_ids':_ids, 'attention_mask':_attention_mask}

	x = transformer.bert(__in)[0]

	global_pool = GlobalAveragePooling1D()(x)

	__out = Dense(__num_classes, activation='sigmoid', name='output')(global_pool)

	model = Model(inputs=__in, outputs=__out)
	model.compile(loss='binary_crossentropy',
				  optimizer=Adam(lr=1e-5),
				  metrics=['accuracy', f1])
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
				  metrics=['accuracy', f1])
	return model


def bert_model2(transformer, max_length=64) -> Model:
	_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')

	x = transformer(_ids)[0]
	x = x[:, 0, :] # cls token
	
	__out = Dense(1, activation='sigmoid', name='output')(x)

	model = Model(inputs=_ids, outputs=__out)
	model.compile(Adam(lr=1e-5),
				  loss='binary_crossentropy',
				  metrics=['accuracy', f1])

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
				  metrics=['accuracy', f1])

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
				  metrics=['accuracy',f1])
	
	return model
