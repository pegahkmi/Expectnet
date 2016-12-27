__author__ = 'kazjon'

import logging
import gensim
import os
import sys
import cPickle as pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import msle as msle_metric
from keras.objectives import msle
from scipy.stats import fisher_exact
import random

def get_surprise(expectnet,i_feature=None,i_context=None,input_pairs=None):
	if input_pairs is not None:
		X = np.stack([expectnet_input_vector(i,j,w2v_model,word_index) for i,j in input_pairs])
	elif i_feature is not None and i_context is not None:
		X = np.stack([expectnet_input_vector(i,j,w2v_model,word_index) for i,j in np.vstack((np.atleast_1d(i_feature),np.atleast_1d(i_context)))])
	else:
		sys.exit("Must provide get_surprise() with either a list of input pairs or individual lists of i_feature and i_context")
	return expectnet.predict(X)

def get_surprise_from_data(i_feature,i_context,corrcounts,n_docs):
	if type(corrcounts) is dict:
		return calc_interpolated_log_conditional_pair(corrcounts[i_feature][i_feature],corrcounts[i_context][i_context],(corrcounts[i_feature][i_context] if i_context in corrcounts[i_feature].keys() else 0),n_docs)
	else:
		return calc_interpolated_log_conditional_pair(corrcounts[i_feature,i_feature],corrcounts[i_context,i_context],corrcounts[i_feature,i_context],n_docs)

def calc_interpolated_log_conditional_pair(n_feature, n_context, n_feature_and_context, n_docs):
	n_feature = float(n_feature)
	n_context = float(n_context)
	n_feature_and_context = float(n_feature_and_context)
	n_docs = float(n_docs)
	marginal_prob = (n_feature+1)/(n_docs+2) # Constants are Laplace's correction (additive smoothing)
	conditional_prob = (n_feature_and_context+1)/(n_context+2)
	pvalue = _calc_significance_of_conditional(n_docs,n_feature,n_feature_and_context,n_context)
	return np.log2((1-pvalue) * conditional_prob + pvalue * marginal_prob)

def _calc_significance_of_conditional(n_docs,n_docs_with_f,n_docs_with_f_and_c,n_docs_with_c):
	return fisher_exact([[n_docs_with_f,n_docs-n_docs_with_f],[n_docs_with_f_and_c,n_docs_with_c-n_docs_with_f_and_c]])[1]

def expectnet_input_vector(i_context,i_feature,w2v_model,word_index):
	return np.concatenate([w2v_model[word_index[i_context]],w2v_model[word_index[i_feature]]])

def gen_training_set(w2v_model, corrcounts, word_index, batch_size, n_docs, indices_to_exclude = []):
	valid_indices = [i for i in range(len(word_index)) if i not in indices_to_exclude]
	batch_indices = np.stack((np.random.choice(len(valid_indices),batch_size),np.random.choice(len(valid_indices)-1,batch_size)),axis=1).tolist()
	batch_indices = [(valid_indices[i],valid_indices[j]) if i<j else (valid_indices[i],valid_indices[j+1]) for i,j in batch_indices]
	X = np.stack([expectnet_input_vector(i,j,w2v_model,word_index) for i,j in batch_indices])
	y = np.stack([get_surprise_from_data(i,j,corrcounts,n_docs) for i,j in batch_indices])
	return X,y

def yield_training_set(path,batch_size,indices_to_exclude = []):
	w2v_model,corrcounts,word_index,n_docs,total_words = load_w2v_and_surp(path)
	valid_indices = [i for i in range(len(word_index)) if i not in indices_to_exclude]
	while 1:
		batch_indices = np.stack((np.random.choice(len(valid_indices),batch_size),np.random.choice(len(valid_indices)-1,batch_size)),axis=1).tolist()
		batch_indices = [(valid_indices[i],valid_indices[j]) if i<j else (valid_indices[i],valid_indices[j+1]) for i,j in batch_indices]
		X = np.stack([expectnet_input_vector(i,j,w2v_model,word_index) for i,j in batch_indices])
		y = np.stack([get_surprise_from_data(i,j,corrcounts,n_docs) for i,j in batch_indices])
		yield X,y


def iterate_training_set(path,batch_size,indices_to_exclude = [],randomise=True):
	w2v_model,corrcounts,word_index,n_docs,total_words = load_w2v_and_surp(path)
	valid_indices = [i for i in range(len(word_index)) if i not in indices_to_exclude]
	while 1:
		idx = valid_indices
		if randomise:
			random.shuffle(idx)
		for batch_i1 in [idx[i:i+batch_size] for i in xrange(0,len(idx),batch_size)]:
			for batch_i2 in [idx[j:j+batch_size] for j in xrange(0,len(idx),batch_size)]:
				batch_indices = zip(batch_i1,batch_i2)
				X = np.stack([expectnet_input_vector(i,j,w2v_model,word_index) for i,j in batch_indices])
				y = np.stack([get_surprise_from_data(i,j,corrcounts,n_docs) for i,j in batch_indices])
				yield X,y

def load_w2v_and_surp(infile):
	w2v_model = gensim.models.Word2Vec.load(infile+"w2v.model")
	with open(infile+"data.corrcounts") as f:
		corrcounts,word_index,n_docs,total_words = pickle.load(f)
	return w2v_model,corrcounts,word_index,n_docs,total_words

def split_dataset(word_index,train_fraction=0.7,val_fraction=0.2,test_fraction=0.1,path=None):
	train_indices = sorted(random.sample(xrange(len(word_index)),int(train_fraction*len(word_index))))
	test_and_val_indices = [i for i in xrange(len(word_index)) if i not in train_indices]
	val_indices = sorted(random.sample(test_and_val_indices,int(val_fraction*len(word_index))))
	test_indices = sorted([i for i in test_and_val_indices if i not in val_indices])
	if path is not None:
		with open(path+"expectnet.split","wb") as f:
			pickle.dump((train_fraction,val_fraction,test_fraction,train_indices,val_indices,test_indices),f)
	return train_indices,val_indices,test_indices

def load_dataset_split(path):
	with open(path+"expectnet.split") as f:
		train_fraction,val_fraction,test_fraction,train_indices,val_indices,test_indices = pickle.load(f)
	return train_fraction,val_fraction,test_fraction,train_indices,val_indices,test_indices

def script_compile_expectnet(layer_sizes,w2v_model_vector_size):
	net_layers = zip([w2v_model_vector_size] + layer_sizes,layer_sizes + [1])
	expectnet = Sequential()
	for i,o in net_layers:
		expectnet.add(Dense(output_dim=o,input_dim=i))
		expectnet.add(Activation("relu"))
	expectnet.compile(loss=msle,optimizer="sgd")#,metrics=[msle_metric])
	return expectnet

#Either data (a train,test tuple of datasets) or path (where the generator can load the data) and train/test indices should be provided.
def script_train_expectnet(expectnet,data=None,path=None,train_indices=[],val_indices=[],test_indices=[],epochs=20,batch_size=10000,sample_size=100000,nb_val_samples=20000,n_cores=4):
	if data is not None:
		X_train,y_train = data
		expectnet.fit(X_train, y_train,nb_epoch=epochs,batch_size=batch_size)
		return expectnet
	if path is not None and len(train_indices):
		train_gen = yield_training_set(path,batch_size,indices_to_exclude=val_indices+test_indices)
		val_gen = yield_training_set(path,batch_size,indices_to_exclude=train_indices+test_indices)
		expectnet.fit_generator(train_gen,sample_size,epochs,validation_data=val_gen,nb_val_samples=nb_val_samples,max_q_size=int(sample_size*epochs/batch_size),pickle_safe=True,nb_worker=n_cores)
		return expectnet
	print "Either data (a train,test dataset tuple) or path (to where the generator can load the data) should be provided. No training was performed."
	return expectnet


#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	path = "ACMDL/"
	layer_sizes = [256,256]
	epochs = 10
	batch_size = 1000
	sample_size = 100000
	train_fraction = 0.7
	val_fraction = 0.2
	test_fraction = 0.1
	generate = True

	w2v_model,corrcounts,word_index,n_docs,total_words = load_w2v_and_surp(path)
	expectnet = script_compile_expectnet(layer_sizes,2*w2v_model.vector_size)

	train_indices,val_indices,test_indices = split_dataset(word_index,path=path)

	if generate:
		expectnet = script_train_expectnet(expectnet,path=path,train_indices=train_indices,val_indices=val_indices,test_indices=test_indices,epochs=epochs,batch_size=batch_size,sample_size=int(sample_size * train_fraction),nb_val_samples=int(sample_size * val_fraction))
	else:
		X_train,y_train = gen_training_set(w2v_model,corrcounts,word_index,sample_size,n_docs,indices_to_exclude=val_indices+test_indices)
		expectnet = script_train_expectnet(expectnet,data=(X_train,y_train),epochs=epochs,batch_size=batch_size)
	print
	X_test,y_test = gen_training_set(w2v_model,corrcounts,word_index,int(sample_size * test_fraction),n_docs,indices_to_exclude=train_indices+val_indices)
	score = expectnet.evaluate(X_test, y_test, batch_size=batch_size)
	expectnet.save(path+"expectnet.model")
	print
	print score