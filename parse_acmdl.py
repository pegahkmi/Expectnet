__author__ = 'kazjon'

import csv,gensim,logging,sys,os.path,multiprocessing
import cPickle as pickle
import numpy as np
import nltk,io
from itertools import repeat,chain
from pathos.multiprocessing import Pool
from nltk.corpus import stopwords,wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from joblib import Parallel, delayed
from scipy.sparse import dok_matrix,hstack
from scipy.stats import fisher_exact
from inflection import singularize


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("expectnet")

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class ACMDL_DocReader(object):
	def __init__(self,path):
		self.filepath = path
		self.stop = set(stopwords.words("english"))
		self.tokeniser = RegexpTokenizer(r'\w+')
		self.correlations = {"___total_words___": 0.0, "___total_docs___": 0.0}
		self.corrs_done = False
		self.finalised = False

	def __iter__(self):
		lem = WordNetLemmatizer()
		with io.open(self.filepath+".csv",mode="r",encoding='ascii',errors="ignore") as i_f:
			for row in csv.DictReader(i_f):
				#Default
				#docwords = [w for w in self.tokeniser.tokenize(row["Abstract"].lower()) if w not in self.stop]

				#Singularise
				docwords = [singularize(w) for w in self.tokeniser.tokenize(row["Abstract"].lower()) if w not in self.stop]

				#tag+lemmatize
				#docwords = nltk.pos_tag(self.tokeniser.tokenize(row["Abstract"].lower()))
				#docwords = [lem.lemmatize(w,pos=get_wordnet_pos(t)) for w,t in docwords if w not in self.stop]
				if not self.corrs_done:
					self.correlations["___total_words___"] += len(docwords)
					unique_docwords = list(set(docwords))
					self.correlations["___total_docs___"] += 1
					for w1 in unique_docwords:
						try:
							self.correlations[w1][w1] += 1
						except KeyError:
							self.correlations[w1] = {w1:1.0}
						#self.correlations["___total_words___"] += 1
						for w2 in unique_docwords:
							if w1 != w2:
								try:
									self.correlations[w1][w2] += 1
								except KeyError:
									self.correlations[w1][w2] = 1.0
				yield docwords
		self.corrs_done = True



	def process(self,suffix=".preprocessed"):
		preprocessed_path = self.filepath+suffix
		if not os.path.exists(preprocessed_path) or not os.path.exists(preprocessed_path+"_corrs"):
			logger.info(" ** Pre-processing started.")
			with open(preprocessed_path,"wb") as pro_f:
				writer = csv.writer(pro_f)
				for doc in self:
					writer.writerow(doc)
			with open(preprocessed_path+"_corrs","wb") as corr_f:
				pickle.dump(self.correlations,corr_f)
			logger.info(" ** Pre-processing complete.")
		else:
			logger.info(" ** Pre-existing pre-processed file found.  Remove "+preprocessed_path+
						" and re-run if you did not intend to reuse it.")
			with open(preprocessed_path+"_corrs","rb") as corr_f:
				self.correlations = pickle.load(corr_f)


	def finalise(self, w2v, num_cores = 12):
		logger.info("Finalising vocab.")
		self.total_words = self.correlations["___total_words___"]
		self.total_docs = self.correlations["___total_docs___"]
		del self.correlations["___total_words___"]
		del self.correlations["___total_docs___"]

		words_to_del = []
		for w in self.correlations.keys():
			if w not in w2v.wv.index2word and not w == "___total_words___":
				self.total_words -= self.correlations[w][w]
				words_to_del.append(w)
		words_to_del = set(words_to_del)
		logger.info(" ** Gathered words to prune.")

		self.correlations = {k:v for k,v in self.correlations.iteritems() if k not in words_to_del}
		self.word_index = self.correlations.keys()
		self.correlations_list = [self.correlations[w] for w in self.word_index]
		logger.info(" ** Completed pruning keys.")

		del self.correlations
		self.corrs_final = hstack(Parallel(n_jobs=num_cores, max_nbytes=1e9)(
									 delayed(prune_to_array)
									 	(corr_subdict,self.word_index) for corr_subdict in self.correlations_list)
								 ).tocsr()
		del self.correlations_list
		logger.info(" ** Completed pruning values. Final corrs shape:"+str(self.corrs_final.shape))

		#self.corrs_dicts = Parallel(n_jobs=num_cores)(delayed(reindex)(self.correlations[w],self.word_index) for w in self.word_index)
		#counts = [float(self.correlations[w][w]) for w in self.word_index]
		logger.info(" ** Completed corrdict calculation.")

		#self.odds_list = Parallel(n_jobs=num_cores, max_nbytes=1e8)(delayed(calc_interpolated_log_conditional)(i,self.total_docs,corrs_dicts[i],counts) for i in word_indices)
		#logger.info(" ** Completed surprise calculation.")

		self.finalised = True

	def save(self, suffix=".corrcounts"):
		if self.finalised:
			with open(self.filepath+suffix,"wb") as corr_f:
				pickle.dump((self.corrs_final,self.word_index,self.total_docs,self.total_words),corr_f)
		else:
			print "Not finalised, refusing to save."
			sys.exit()

	def train_w2v(self,suffix=".w2vmodel",size=64,min_count=25,iter=25,num_cores=12):
		fast_reader = W2VReader(self.filepath)
		self.model = gensim.models.Word2Vec(fast_reader,workers=num_cores,min_count=min_count,iter=iter,size=size)
		self.model.save(self.filepath+suffix)
		return self.model


class W2VReader(object):
	def __init__(self,path,suffix=".preprocessed"):
		self.filepath = path+suffix

	def __iter__(self):
		with open(self.filepath,"rb") as i_f:
			for row in csv.reader(i_f):
				yield row

def prune(key,corr_subdict,words_to_del):
	return (key,{w:c for w,c in corr_subdict.iteritems() if w not in words_to_del})

def prune_to_array(corr_subdict,word_index):
	corr_array = dok_matrix((len(word_index),1), dtype=np.float32)
	for i,w in enumerate(word_index):
		try:
			corr_array[i] = corr_subdict[w]
		except KeyError:
			pass
	return corr_array.tocsr()

def reindex(corr_dict, word_index): 
	return {word_index.index(w):v for w,v in corr_dict.iteritems()}


if __name__ == "__main__":
	inputfile = sys.argv[1]
	path = "acmdl/"
	acm = ACMDL_DocReader(os.path.join(path,inputfile))
	acm.process()
	model = acm.train_w2v(min_count=10,iter=50, num_cores=multiprocessing.cpu_count())
	acm.finalise(model, num_cores=multiprocessing.cpu_count())
	acm.save()
	print model.most_similar("computer")
	print model.most_similar("research")
