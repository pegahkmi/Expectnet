#!/bin/sh

virtualenv venv
source venv/bin/activate
pip install numpy scipy pattern theano keras joblib gensim inflection pathos nltk
