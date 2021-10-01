import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


def displot_tokenized_sentences(texts , pad_token=0):
	len_sentences_tokenized = [np.count_nonzero(txt != pad_token) for txt in texts]
	_where_to_cut = np.mean(len_sentences_tokenized) + np.std(len_sentences_tokenized)

	sns.displot(len_sentences_tokenized)
	plt.title('Sentence length, cut at +1 SD  (%d)' % int(_where_to_cut), pad=1)
	plt.axvline(_where_to_cut, color='r')
	plt.show()


def find_best_classification_threshold(y_true, probability_predictions, step=0.05, verbose=True):
	start, end = np.min(probability_predictions), np.max(probability_predictions) 
	best_cuttof = 0
	best_score = -99999.0

	_x = []
	_y = []
	current = start
	while current <= end:
		_y_pred_new = [0 if y < current else 1 for y in probability_predictions]
		_f1 = f1_score(y_true, _y_pred_new, average='macro')
		if _f1 > best_score:
			best_score = _f1
			best_cuttof = round(current, 3)
		_x.append(current)
		_y.append(_f1)
		current += step

	if verbose:
		print('threshold f1:', round(best_score, 3), '  @cut <', best_cuttof)

		plot_range_threshold = 0.94 * best_score

		for xi, yi in zip(_x.copy(), _y.copy()):
			if yi < plot_range_threshold:
				_x.remove(xi)
				_y.remove(yi)

		sns.set()
		#plt.figure(figsize=(10, 5), dpi=220)
		plt.plot(_x, _y)
		plt.title('F1 per clf/ion Threshold')
		plt.plot(best_cuttof, best_score, 'r+')
		plt.xticks(np.linspace(min(_x), max(_x), num=25), rotation=50, size=10)
		plt.show()

	return best_cuttof

def convert_logits_to_probability(logits):
	return [logit[1] - logit[0] for logit in logits]

def evaluate_model(model, x_val, y_val, x_test, y_test):
	# 1 predict probabilities for x_val
	# 2 find best classification threshold for probabilities
	# 3 predict x_test --> y_pred_proba
	# 4 apply that threshold to probabilities --> y_pred
	# 5 evaluate y_pred on various metrics 

	tune_proba = True
	threshold = 0

	# 1
	if 'predict_proba' in dir(model):
		y_pred = model.predict_proba(x_val)
	else:
		y_pred = model.predict(x_val)
	
	if 'logits' in dir(y_pred):  # each prediction is 2 logits
		y_pred = convert_logits_to_probability(y_pred.logits)

	if type(y_pred[0]) in [np.int64, np.int32, int]:
		tune_proba = False

	# 2
	if tune_proba:
		threshold = find_best_classification_threshold(y_val, y_pred)

		# 3
		if 'predict_proba' in dir(model):
			y_pred = model.predict_proba(x_test)
		else:
			y_pred = model.predict(x_test)

		if 'logits' in dir(y_pred):  # each prediction is 2 logits
			y_pred = convert_logits_to_probability(y_pred.logits)
	# 4
	y_pred_classes = [0 if y < threshold else 1 for y in y_pred]

	# 5
	f1 = f1_score(y_test, y_pred_classes, average='macro')
	acc = accuracy_score(y_test, y_pred_classes)
	roc_auc = roc_auc_score(y_test, y_pred_classes, average='macro')
	p = precision_score(y_test, y_pred_classes, average='macro')
	r = recall_score(y_test, y_pred_classes, average='macro')

	print('\n[---------------------  %s  ---------------------]\nf1=%.3f | acc=%.3f | rocAUC=%.3f | p=%.3f | rec=%.3f'
		  % (model.name if 'name' in dir(model) else model.__class__.__name__, f1, acc, roc_auc, p, r))


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


# other

# models to test
# https://arxiv.org/pdf/1912.06872.pdf		#Towards Robust Toxic Content Classification

# bert implementations																	#notes
# https://www.kaggle.com/nayansakhiya/text-classification-using-bert					simple
# https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-with-huggingface-and-keras			tf.dataset | fast encode | cls token
# https://www.kaggle.com/gtskyler/toxic-comments-bert/									average pooling
# https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras		distilbert | tf.dataset  | cls token
# https://www.kaggle.com/harveenchadha/chaii-tpu-train-nfold-xlm-hf-tf-data-extra		xlmr | conv on bert

# https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671
# https://www.tensorflow.org/text/tutorials/fine_tune_bert
# https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22
# https://www.analyticsvidhya.com/blog/2020/10/simple-text-multi-classification-task-using-keras-bert/
# https://medium.com/analytics-vidhya/bert-in-keras-tensorflow-2-0-using-tfhub-huggingface-81c08c5f81d8

# embeddings
# https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b
# http://archive.aueb.gr:7000/resources/
