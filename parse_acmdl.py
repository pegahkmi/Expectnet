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
inputfile = "ACMDL/PIQUE_Research papers_v1.0.csv"

'''
def calc_word_pmis(args):
	return _calc_word_pmis(*args)

def _calc_word_pmis(i_context, num_docs, corrs, counts):
		n_context = counts[i_context]
		p_context = n_context/num_docs
		pmi_dict = {}
		#pmi_row = np.zeros(num_words)
		for i_feature,n_feature_and_context in corrs.iteritems():
			if i_feature != i_context:
				n_feature = counts[i_feature]
				p_feature = n_feature/num_docs
				n_feature_and_context = n_feature_and_context/num_docs
				#pmi_row[i_feature] = np.log2(p_w1w2/(p_w1*p_w2))
				pmi_dict[i_feature] = np.log2(n_feature_and_context/(p_context*p_feature))
		return pmi_dict
		#return pmi_row

def _calc_word_odds_ratio(i_context,num_docs,corrs,counts, bayes=True):
	p_context = counts[i_context]/num_docs
	odds_dict = {}
	for i_feature,n_feature_and_context in corrs.iteritems():
		if i_feature != i_context:
			if bayes:
				p_feature = counts[i_feature]/num_docs
				p_context_given_feature = (n_feature_and_context+1)/(counts[i_context]+2) #Constants are Laplace's
				p_feature_given_context = (p_context_given_feature/p_context) * p_feature
			else:
				p_feature_given_context = counts[i_feature]/num_docs
			odds_dict[i_feature] = np.log2(p_feature_given_context/p_context)
	return odds_dict
'''

'''
def calc_interpolated_log_conditional(i_context,num_docs,corrs,counts):
	log_conditionals = {}
	for i_feature,n_feature_and_context in corrs.iteritems():
		if i_feature != i_context:
			log_conditionals[i_feature] = calc_interpolated_log_conditional_pair(counts[i_feature], counts[i_context],n_feature_and_context,num_docs)
	return log_conditionals

def calc_interpolated_log_conditional_pair(n_feature, n_context, n_feature_and_context, n_docs):
	marginal_prob = (n_feature+1)/(n_docs+2) # Constants are Laplace's correction (additive smoothing)
	conditional_prob = (n_feature_and_context+1)/(n_context+2)
	pvalue = _calc_significance_of_conditional(n_docs,n_feature,n_feature_and_context,n_context)
	return np.log2(pvalue * conditional_prob + (1-pvalue) * marginal_prob)

def _calc_significance_of_conditional(n_docs,n_docs_with_f,n_docs_with_f_and_c,n_docs_with_c):
	return fisher_exact([[n_docs_with_f,n_docs-n_docs_with_f],[n_docs_with_f_and_c,n_docs_with_c-n_docs_with_f_and_c]])[1]
'''

class ACMDL_SentenceReader(object):
	def __init__(self,path):
		self.filepath = path
		self.stop = stopwords.words("english")
		self.tokeniser = RegexpTokenizer(r'\w+')
		self.correlations = {"___total_words___": 0.0}
		self.corrs_done = False

	def __iter__(self):
		with open(self.filepath,"rb") as i_f:
			for row in csv.DictReader(i_f):
				if not self.corrs_done:
					docwords = [w for w in self.tokeniser.tokenize(row["Abstract"].lower()) if w not in self.stop]
				for s in row["Abstract"].split("."):
					r =  [w for w in self.tokeniser.tokenize(s.lower()) if w not in self.stop]
					if not self.corrs_done:
						for w1 in r:
							if w1 not in self.correlations.keys():
								self.correlations[w1] = {}
								self.correlations[w1][w1] = 0.0
							self.correlations[w1][w1] += 1
							self.correlations["___total_words___"] += 1
							for w2 in r: #Switching this between r and docwords switches the corrtab between per-sentence and per-abstract counts.
								if w1 != w2:
									if w2 not in self.correlations[w1].keys():
										self.correlations[w1][w2] = 0.0
									self.correlations[w1][w2] += 1
					yield r
		'''
		if not self.corrs_done: #Now we have the entire vocab, go through the dictionary and say that words which aren't in there already occurred alongside each other 0 times.
			for w1 in self.correlations.keys():
				if w1 != "___total_words___":
					for w2 in self.correlations.keys():
						if w2 != "___total_words___":
							if w2 not in self.correlations[w1].keys():
								self.correlations[w1][w2] = 0.0
		else:
			self.corrs_done = True
		'''
		self.corrs_done = True

class ACMDL_DocReader(object):
	def __init__(self,path):
		self.filepath = path
		self.stop = set(stopwords.words("english"))
		self.tokeniser = RegexpTokenizer(r'\w+')
		self.correlations = {"___total_words___": 0.0, "___total_docs___": 0.0}
		self.corrs_done = False
		self.finalised = False

	def __iter__(self):
		with open(self.filepath,"rb") as i_f:
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

	def process(self):
		logger.info(" ** Pre-processing started.")
		with open(self.filepath[:-4]+".preprocessed","wb") as pro_f:
			writer = csv.writer(pro_f)
			for doc in self:
				writer.writerow(doc)
		logger.info(" ** Pre-processing complete.")

	def finalise(self, w2v, num_cores = 12):
		logger.info("Finalising vocab.")
		self.total_words = self.correlations["___total_words___"]
		self.total_docs = self.correlations["___total_docs___"]
		del self.correlations["___total_words___"]
		del self.correlations["___total_docs___"]

		words_to_del = []
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

	def train_w2v(self,outputfile,size=64,min_count=25,iter=25,num_cores=12):
		self.model = gensim.models.Word2Vec(W2VReader(self.filepath),workers=num_cores,min_count=min_count,iter=iter,size=size)
		self.model.save(outputfile+"w2v.model")
		return self.model


class W2VReader(object):
	def __init__(self,path,suffix="data.preprocessed"):
		self.filepath = os.path.join(path,suffix)

	def __iter__(self):
		with open(self.filepath,"rb") as i_f:
			for row in csv.reader(i_f):
				yield row

def prune(key,corr_subdict,words_to_del):
	return (key,{w:c for w,c in corr_subdict.iteritems() if w not in words_to_del})

def prune_to_array(corr_subdict,word_index):
	return np.array([corr_subdict[w] for w in word_index])

def reindex(corr_dict, word_index): 
	return {word_index.index(w):v for w,v in corr_dict.iteritems()}


if __name__ == "__main__":
	inputfile = "ACMDL_complete.csv"
	path = "acmdl/"
	acm = ACMDL_DocReader(os.path.join(path,inputfile))
	acm.process()
	model = acm.train_w2v(path)
	acm.finalise(model)
	acm.save(path)
	print model.most_similar("computer")
	print model.most_similar("research")
