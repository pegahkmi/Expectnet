__author__ = 'kazjon'

import csv,gensim,logging,sys,os.path
import cPickle as pickle
import numpy as np
from itertools import repeat,chain
from pathos.multiprocessing import Pool
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.stats import fisher_exact

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("expectnet")

class ACMDL_DocReader(object):
	def __init__(self,path):
		self.filepath = path
		self.stop = set(stopwords.words("english"))
		self.tokeniser = RegexpTokenizer(r'\w+')
		self.correlations = {"___total_words___": 0.0, "___total_docs___": 0.0}
		self.corrs_done = False
		self.finalised = False

	def __iter__(self):
		with open(self.filepath+".csv","rb") as i_f:
			for row in csv.DictReader(i_f):
				docwords = [w for w in self.tokeniser.tokenize(row["Abstract"].lower()) if w not in self.stop]
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
		if not os.path.exists(preprocessed_path):
			logger.info(" ** Pre-processing started.")
			with open(preprocessed_path,"wb") as pro_f:
				writer = csv.writer(pro_f)
				for doc in self:
					writer.writerow(doc)
			with open(preprocessed_path+"_corrs","wb") as corr_f:
				pickle.dump(self.correlations,corr_f)
			logger.info(" ** Pre-processing complete.")
		else:
			logger.info(" ** Pre-existing pre-processed file found.  Remove "+preprocessed_path+" and re-run if you did not intend to reuse it.")
			with open(preprocessed_path+"_corrs","rb") as corr_f:
				self.correlations = pickle.load(corr_f)


	def finalise(self, w2v, num_cores = 12):
		logger.info("Finalising vocab.")
		self.total_words = self.correlations["___total_words___"]
		self.total_docs = self.correlations["___total_docs___"]
		del self.correlations["___total_words___"]
		del self.correlations["___total_docs___"]

		words_to_del = []
		print self.correlations
		for w in self.correlations.keys():
			if w not in w2v.index2word and not w == "___total_words___":
				self.total_words -= self.correlations[w][w]
				words_to_del.append(w)
		words_to_del = set(words_to_del)
		logger.info(" ** Gathered words to prune.")

		#self.correlations = {k:{w:c for w,c in v.iteritems() if w not in words_to_del} for k,v in self.correlations.iteritems() if k not in words_to_del}
		self.correlations = {k:v for k,v in self.correlations.iteritems() if k not in words_to_del}
		self.word_index = self.correlations.keys()
		self.correlations_list = [self.correlations[w] for w in self.word_index]
		logger.info(" ** Completed pruning keys.")

		#self.correlations = dict(Parallel(n_jobs=num_cores, max_nbytes=1e9)(delayed(prune)(key,corr_subdict,words_to_del) for key,corr_subdict in self.correlations.iteritems()))
		self.corrs_final = np.vstack(Parallel(n_jobs=num_cores, max_nbytes=1e9)(delayed(prune_to_array)(corr_subdict,self.word_index) for corr_subdict in self.correlations_list))
		#for k,v in self.correlations.iteritems():
		#	self.correlations[k] = {w:c for w,c in v.iteritems() if w not in words_to_del}
		logger.info(" ** Completed pruning values.")

		#self.corrs_dicts = Parallel(n_jobs=num_cores)(delayed(reindex)(self.correlations[w],self.word_index) for w in self.word_index)
		#counts = [float(self.correlations[w][w]) for w in self.word_index]
		logger.info(" ** Completed corrdict calculation.")

		#self.odds_list = Parallel(n_jobs=num_cores, max_nbytes=1e8)(delayed(calc_interpolated_log_conditional)(i,self.total_docs,corrs_dicts[i],counts) for i in word_indices)
		#logger.info(" ** Completed surprise calculation.")

		self.finalised = True

	def save(self, out_fn):
		if self.finalised:
			with open(out_fn+"data.corrcounts","wb") as corr_f:
				pickle.dump((self.corrs_final,self.word_index,self.total_docs,self.total_words),corr_f)
		else:
			print "Not finalised, refusing to save."
			sys.exit()

	def train_w2v(self,outputpath,size=64,min_count=25,iter=25,num_cores=12):
		self.model = gensim.models.Word2Vec(W2VReader(self.filepath),workers=num_cores,min_count=min_count,iter=iter,size=size)
		self.model.save(outputpath+"w2v.model")
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
	corr_array = np.zeros(len(word_index))
	for i,w in enumerate(word_index):
		try:
			corr_array[i] = corr_subdict[w]
		except KeyError:
			pass
	return corr_array

def reindex(corr_dict, word_index): 
	return {word_index.index(w):v for w,v in corr_dict.iteritems()}


if __name__ == "__main__":
	inputfile = sys.argv[1]
	path = "acmdl/"
	acm = ACMDL_DocReader(os.path.join(path,inputfile))
	acm.process()
	model = acm.train_w2v(path,min_count=100,iter=100)
	acm.finalise(model)
	acm.save(path)
	print model.most_similar("computer")
	print model.most_similar("research")
