import csv, gensim, logging, sys, os.path, multiprocessing, nltk, io, glove, argparse, glob
import cPickle as pickle
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from joblib import Parallel, delayed
from scipy.sparse import dok_matrix, hstack
from inflection import singularize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("Expectnet")


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


class DocReader(object):
    def __init__(self, path):
        self.filepath = path
        self.total_words = 0
        self.total_docs = 0
        self.stop = set(stopwords.words("english"))
        self.tokeniser = RegexpTokenizer(r'\w+')
        self.first_pass = True
        self.finalised = False
        self.doc_ids = []
        self.doc_titles = []
        self.doc_raws = []
        self.lem = WordNetLemmatizer()

    def __iter__(self):
        raise NotImplementedError

    def load(self, preprocessed_path):
        with open(preprocessed_path, "rb") as pro_f:
            self.documents, self.word_occurrence, self.cooccurrence, self.dictionary, self.total_docs, self.doc_ids, self.doc_titles, self.doc_raws = pickle.load(
                pro_f)
            self.first_pass = False

    def preprocess(self, suffix=".preprocessed", no_below=0.001, no_above=0.5, force_overwrite=False, size=64,
                   min_count=25, iter=25, num_cores=2):
        """construct bag of words and co-occurance matrix"""
        self.argstring = "_below" + str(no_below) + "_above" + str(no_above)
        preprocessed_path = self.filepath + self.argstring + suffix
        if not os.path.exists(preprocessed_path) or force_overwrite:
            logger.info(" ** Pre-processing started.")
            self.dictionary = gensim.corpora.Dictionary(self)
            logger.info("   **** Dictionary created.")
            self.dictionary.filter_extremes(no_below=max(2, no_below * self.total_docs), no_above=no_above, keep_n=None)
            self.word_occurrence = {k: 0.0 for k in self.dictionary.token2id.keys()}
            logger.info("   **** Dictionary filtered.")
            self.documents = [self.dictionary.doc2bow(d) for d in self]
            logger.info("   **** BoW representations constructed.")
            self.calc_cooccurrence()
            logger.info("   **** Co-occurrence matrix constructed.")
            self.w2v_model = gensim.models.Word2Vec(self, workers=num_cores, min_count=min_count, iter=iter, size=size)

            with open(preprocessed_path, "wb") as pro_f:
                pickle.dump((self.documents, self.word_occurrence, self.cooccurrence, self.dictionary, self.total_docs,
                             self.doc_ids, self.doc_titles, self.doc_raws, self.w2v_model), pro_f)
        else:
            logger.info(" ** Existing pre-processed file found.  Rerun with --overwrite_preprocessing" +
                        " if you did not intend to reuse it.")
            self.load(preprocessed_path)
        logger.info(" ** Pre-processing complete.")

    def calc_cooccurrence(self, normalise=False):
        """co-occurance matrix"""
        self.cooccurrence = {wk: {} for wk in range(len(self.dictionary))}
        for doc in self.documents:
            self.total_docs += 1.0
            for wk, wc in doc:
                self.total_words += wc
                self.word_occurrence[self.dictionary[wk]] += 1.0
                for wk2, wc2 in doc:
                    if wk != wk2:
                        try:
                            self.cooccurrence[wk][wk2] += 1.0
                        except KeyError:
                            self.cooccurrence[wk][wk2] = 1.0
        if normalise:
            for wk, wv in self.dictionary.iteritems():
                self.cooccurrence[wk] = {wk2: float(wv2) / self.word_occurrence[wv] for wk2, wv2 in
                                         self.cooccurrence[wk].iteritems()}

    def save(self, filepath, suffix=".corrcounts"):
        word_index = self.cooccurrence.keys()
        correlations_list = [self.cooccurrence[w] for w in word_index]
        self.corrs_final = hstack(Parallel(n_jobs=12, max_nbytes=1e9)(
            delayed(prune_to_array)
            (corr_subdict, word_index) for corr_subdict in correlations_list)
        ).tocsr()
        with open(filepath + suffix, "wb") as corr_f:
            pickle.dump((self.corrs_final, word_index, self.total_docs, self.total_words), corr_f)


def prune(key, corr_subdict, words_to_del):
    return (key, {w: c for w, c in corr_subdict.iteritems() if w not in words_to_del})


def prune_to_array(corr_subdict, word_index):
    corr_array = dok_matrix((len(word_index), 1), dtype=np.float32)
    for i, w in enumerate(word_index):
        try:
            corr_array[i] = corr_subdict[w]
        except KeyError:
            pass
    return corr_array.tocsr()


class ACMDL_ExpectnetDocReader(DocReader):
    def __init__(self, path):
        DocReader.__init__(self, path)

    def __iter__(self):
        with io.open(self.filepath + ".csv", mode="r", encoding='ascii', errors="ignore") as i_f:
            for row in csv.DictReader(i_f):

                docwords = [singularize(w) for w in
                            self.tokeniser.tokenize((row["Abstract_Title"] + " " + row["Abstract"]).lower()) if
                            w not in self.stop]

                if self.first_pass:
                    self.total_words += len(docwords)
                    self.total_docs += 1
                    self.doc_ids.append(row["ID"])
                    self.doc_titles.append(row["Abstract_Title"])
                    self.doc_raws.append(row["Abstract"])

                yield docwords
        self.first_pass = False


class WikiPlot_DocReader(DocReader):
    def __init__(self, path):
        DocReader.__init__(self, path)

    def __iter__(self):
        with io.open(self.filepath, mode="r", encoding='ascii', errors="ignore") as i_f:
            if self.first_pass:
                doc_raw = ""
                t_f = io.open(self.filepath + "_titles", mode="r", encoding='ascii', errors="ignore")
            docwords = []
            for line in i_f.readlines():
                if line[:5] == "<EOS>":
                    if self.first_pass:
                        self.total_words += len(docwords)
                        self.doc_ids.append(self.total_docs)
                        self.total_docs += 1
                        self.doc_titles.append(t_f.readline())
                        self.doc_raws.append(doc_raw)
                        doc_raw = ""
                    yield docwords
                    docwords = []
                else:
                    docwords += [singularize(w) for w in self.tokeniser.tokenize(line) if w not in self.stop]
                    if self.first_pass:
                        doc_raw += line
        if self.first_pass:
            t_f.close()
        self.first_pass = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Word2vec on some text.")
    parser.add_argument("inputfile", help='The file path to work with (omit the ".csv")')
    parser.add_argument("--dataset", default="acm", type=str,
                        help="Which dataset to assume.  Currently 'acm' or 'plots'")
    parser.add_argument("--size", default=64, type=int, help="The size of w2v model.")
    parser.add_argument("--min_count", default=25, type=int, help="The number of dimensions in the GloVe vectors.")
    parser.add_argument("--iter", default=25, type=int, help="The number of dimensions in the GloVe vectors.")
    parser.add_argument("--num_cores", default=multiprocessing.cpu_count(), type=int, help="The number of cores.")
    parser.add_argument("--no_below", default=0.001, type=float,
                        help="Min fraction of documents a word must appear in to be included.")
    parser.add_argument("--no_above", default=0.75, type=float,
                        help="Max fraction of documents a word can appear in to be included.")
    parser.add_argument("--overwrite_model", action="store_true",
                        help="Ignore (and overwrite) existing .w2v file.")
    parser.add_argument("--overwrite_preprocessing", action="store_true",
                        help="Ignore (and overwrite) existing .preprocessed file.")

    args = parser.parse_args()
    if args.dataset == "acm":
        reader = ACMDL_ExpectnetDocReader(args.inputfile)
    elif args.dataset == "plots":
        reader = WikiPlot_DocReader(args.inputfile)
    else:
        logger.info("You've tried to load a dataset we don't know about.  Sorry.")
        sys.exit()
    reader.preprocess(no_below=args.no_below, no_above=args.no_above, force_overwrite=args.overwrite_preprocessing,
                      size=args.size, min_count=args.min_count, iter=args.iter, num_cores=args.num_cores)
