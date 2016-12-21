__author__ = 'kazjon'

import keras.models
import logging

from train_expectnet import load_w2v_and_surp,load_dataset_split, gen_training_set

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	path = "ACMDL/"

	batch_size = 1000
	test_sample_size = 100000

	w2v_model,corrcounts,word_index,n_docs,total_words = load_w2v_and_surp(path)
	expectnet = keras.models.load_model(path+"expectnet.model")
	train_fraction,val_fraction,test_fraction,train_indices,val_indices,test_indices = load_dataset_split(path)

	X_test,y_test = gen_training_set(w2v_model,corrcounts,word_index,test_sample_size,n_docs,indices_to_exclude=train_indices+val_indices)
	score = expectnet.evaluate(X_test, y_test, batch_size=batch_size)
	print
	print score
