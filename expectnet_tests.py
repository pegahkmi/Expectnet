__author__ = 'kazjon'

import unittest,keras,os.path,csv,string,math
import numpy as np
from parse_acmdl import ACMDL_DocReader
from train_expectnet import load_w2v_and_surp,script_compile_expectnet,calc_interpolated_log_conditional_pair,split_dataset,script_train_expectnet,gen_training_set,expectnet_input_vector


'''
class Test_parse_acmdl(unittest.TestCase):

	@classmethod
	def setup_class(self):
		self.inputfile = "ACMDL/PIQUE_Research papers_v1.0_pruned.csv"
		self.path = "ACMDL/"
		self.acm = ACMDL_DocReader(self.inputfile)
		self.model = self.acm.train_w2v(self.path,min_count=100,iter=2)

	@classmethod
	def teardown_class(self):
		pass

	#The user should be able to specify a dataset on which to train expectnet, and receive a trained w2v model file.
	def test_w2v_saved(self):
		w2v_model,_,_,_,_ = load_w2v_and_surp(self.path)
		print self.model.syn0
		print w2v_model.syn0
		self.assertTrue(all([np.allclose(i,j) for i,j in zip(self.model.syn0,w2v_model.syn0)]))


	#The user should be able to specify a dataset on which to train expectnet, and receive a correlation counts file.
	def test_corrcounts_saved(self):
		self.acm.finalise(self.model)
		self.acm.save(self.path)
		w2v_model,corrcounts,word_index,n_docs,total_words = load_w2v_and_surp(self.path)
		self.assertEqual(self.acm.corrs_dicts,corrcounts)


'''
class Test_w2v_and_Expectnet(unittest.TestCase):


	@classmethod
	def gen_letter_chances(self,odds_mat,context):
		p = np.ones(26)
		for c in context:
			p = np.multiply(p,odds_mat[string.ascii_lowercase.index(c),:])
		return np.array([i/np.sum(p) for i in p])

	@classmethod
	def generate_letter_test_data(self,fn="letter_testdata.csv",n_docs=1000,doc_length=100,window=5,odds_max=3):
		with open(os.path.join(self.path,fn),"wb") as f:
			writer = csv.writer(f)
			writer.writerow(["Abstract",""])
			doc = []
			odds_max -= 1
			odds_mat = np.random.random([26,26])*(2*(odds_max))-odds_max
			odds_mat = np.reshape(np.array([1/(-o+1) if o < 0 else o+1 for o in np.nditer(odds_mat)]),[26,26])
			odds_mat = np.triu(odds_mat)
			odds_mat = odds_mat + odds_mat.T - np.diag(odds_mat.diagonal())
			np.fill_diagonal(odds_mat,1)
			rows = 0
			while rows < n_docs:
				doc = []
				while len(doc) < doc_length:
					dist = self.gen_letter_chances(odds_mat,doc[:min(len(doc)-1,window)])
					doc.append(np.random.choice(list(string.ascii_lowercase),p=dist))
				writer.writerow([" ".join(doc)]+[""])
				rows +=1
		with open(os.path.join(self.path,"letter_corrmat.csv"),"wb") as f:
			writer = csv.writer(f)
			writer.writerow(["*"]+list(string.ascii_lowercase))
			for i,row in enumerate(odds_mat):
				writer.writerow([string.ascii_lowercase[i]]+list(row))

	@classmethod
	def setUpClass(self):
		self.path = "testing/"
		self.testfn = "letter_testdata.csv"
		self.testpath = os.path.join(self.path,self.testfn)
		self.generate_letter_test_data(fn=self.testfn)

	@classmethod
	def tearDownClass(self):
		pass

	#The trained w2v model file should be able to be loaded and prompted for similarity between words. Words with highly similar distributional semantics should be rated accordingly.
	def test_w2v_similarity(self):
		#Generate some test documents with letters as words and a random letter-letter coocurrence matrix
		#Split the letter a into two randomly: a1 and a2
		docs = []
		with open(self.testpath,"rb") as tf_i:
			reader = csv.reader(tf_i)
			reader.next()
			for row in reader:
				docs.append(row[0])

		for i,doc in enumerate(docs):
			docs[i] = [l if l is not "a" else l+str(int(1+np.random.randint(2))) for l in str.split(doc)]

		mod_testpath = os.path.join(self.path,"letter_split_testdata.csv")
		with open(mod_testpath,"wb") as tf_o:
			writer = csv.writer(tf_o)
			writer.writerow(["Abstract",""])
			for doc in docs:
				writer.writerow([" ".join(doc)]+[""])

		#train w2v
		self.acm = ACMDL_DocReader(mod_testpath)
		self.model = self.acm.train_w2v(self.path,min_count=5,iter=10)
		#verify that a1 and a2 are highly similar
		self.assertAlmostEqual(1,self.model.similarity("a1","a2"),places=1)


	#The interpolated log probability should lean towards the conditional when the two words co-occur frequently and towards the marginal when they do not.
	def test_log_probability_interpolator(self):
		#This should be a true unit test of calc_interpolated_log_conditional_pair, which APPEARS according to the below to be returning nonsense?
		#inputs are: n_feature, n_context, n_feature_and_context, n_docs
		#print math.pow(2,calc_interpolated_log_conditional_pair(100,100,1,1000))
		#print math.pow(2,calc_interpolated_log_conditional_pair(100,1,0,1000))
		self.assertAlmostEqual(0.01,math.pow(2,calc_interpolated_log_conditional_pair(100,100,1,1000)),places=1) #With one hundred examples, this should be pretty sure the conditional (~0.01) is right
		self.assertAlmostEqual(0.1,math.pow(2,calc_interpolated_log_conditional_pair(100,1,0,1000)),places=1) #With only one example, this should be highly not-confident and revert to the marginal (~0.1)

	#We should be able to initialise, save, and load an expectnet model.
	def test_expectnet_initialisation(self):
		expectnet = script_compile_expectnet([256,256],100)
		expectnet.save(self.path+"expectnet.model")
		self.assertEqual(expectnet.to_json(),keras.models.load_model(self.path+"expectnet.model").to_json())

	#A trained expectnet should closely match the interpolated log probability for common and rare words in the training set.
	def test_expectnet_training_error(self):
		#Generate some test documents with letters as words and "Zeno's paradox" frequency, and random letter-letter coocurrence.
		#train w2v
		#train expectnet
		#Test some known log probs from the training data

		epochs = 10
		batch_size = 100
		sample_size = 1000
		train_fraction = 0.7
		val_fraction = 0.2
		test_fraction = 0.1

		acm = ACMDL_DocReader(self.testpath)
		w2v_model = acm.train_w2v(self.path,min_count=5,iter=10)
		acm.finalise(w2v_model)
		acm.save(self.path)
		w2v_model,corrcounts,word_index,n_docs,total_words = load_w2v_and_surp(self.path)

		self.train_indices,self.val_indices,self.test_indices = split_dataset(word_index,path=self.path)

		expectnet = script_compile_expectnet([256,256],100)
		#expectnet = script_train_expectnet(expectnet,path=self.path,train_indices=self.train_indices,val_indices=self.val_indices,test_indices=self.test_indices,epochs=epochs,batch_size=batch_size,sample_size=int(sample_size * train_fraction),nb_val_samples=int(sample_size * val_fraction))

		#X_test,y_test = gen_training_set(w2v_model,corrcounts,word_index,int(sample_size * test_fraction),n_docs,indices_to_exclude=self.train_indices+self.val_indices)
		#score = expectnet.evaluate(X_test, y_test, batch_size=batch_size)
		#expectnet.save(self.path+"expectnet.model")

		n_train_samples = 10
		train_sample_indices = np.stack((np.random.choice(len(self.train_indices),n_train_samples),np.random.choice(len(self.train_indices)-1,n_train_samples)),axis=1).tolist()
		train_sample_indices = [(self.train_indices[i],self.train_indices[j]) if i<j else (self.train_indices[i],self.train_indices[j+1]) for i,j in train_sample_indices]
		train_samples = np.stack([expectnet_input_vector(i,j,w2v_model,word_index) for i,j in train_sample_indices])
		train_sample_probs = np.stack([calc_interpolated_log_conditional_pair(corrcounts[i][i],corrcounts[j][j],(corrcounts[i][j] if j in corrcounts[i].keys() else 0),n_docs) for i,j in train_sample_indices])
		self.assertEqual(train_sample_probs,expectnet.predict(train_samples,verbose=1))

	#A trained expectnet should closely match the interpolated log probability for common and rare words in the test set.
	def test_expectnet_test_error(self):
		#Generate some test documents with letters as words and "Zeno's paradox" frequency, and random letter-letter coocurrence.
		#Split those test documents into training and testing data
		#train w2v
		#train expectnet

		#Test some known log probs from the test data
		self.assertEqual(True,False)

	def test_surprise_eval(self):
		#Generate some test documents with letters as words and "Zeno's paradox" frequency, and random letter-letter coocurrence.
		#Split those test documents into training and testing data
		#train w2v
		#train expectnet

		#Test some known log probs from the test data
		self.assertEqual(True,False)
