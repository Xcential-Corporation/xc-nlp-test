# python
import json
import time
import pickle
import re, string
import os
from os import path, listdir
from pathlib import Path
from os.path import isfile, join
from types import new_class
from typing import List
from lxml import etree 
from contextlib import ExitStack
import sklearn.feature_extraction.text
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer, TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Among the larger bills is samples/congress/116/BILLS-116s1790enr.xml (~ 10MB)

PATH_116_USLM = '/usr/local/share/xcential/public/data/116/uslm'
PATH_117_USLM = '/usr/local/share/xcential/public/data/117/uslm'
PATH_116_USLM_TRAIN = 'samples/congress/116/train'
PATH_116_TEXT = 'samples/congress/116/txt'

BILLS_SAMPLE = [f'BILLS-116hr{number}ih.xml' for number in range(100, 300)]
BIG_BILLS = ['BILLS-116s1790enr.xml', 'BILLS-116hjres31enr.xml']
BIG_BILLS_PATHS = [path.join(PATH_116_USLM, bill) for bill in (BIG_BILLS + BILLS_SAMPLE)]

SAMPLE_BILL_PATHS_TRAIN = [join(PATH_116_USLM_TRAIN, f) for f in listdir(PATH_116_USLM) if isfile(join(PATH_116_USLM_TRAIN, f))]
SAMPLE_BILL_PATHS = [join(PATH_117_USLM, f) for f in listdir(PATH_117_USLM) if isfile(join(PATH_117_USLM, f))]


NAMESPACES = {'uslm': 'http://xml.house.gov/schemas/uslm/1.0'}

def text_cleaning(text):
    """
    Clean text
    """
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def get_filepaths(dirpath: str, reMatch = r'.xml$') -> List[str]:
    return [join(dirpath, f) for f in listdir(dirpath) if (len(re.findall(reMatch, f)) > 0) and isfile(join(dirpath, f))]

def getEnum(section) -> str:
  enumpath = section.xpath('enum')  
  if len(enumpath) > 0:
    return enumpath[0].text
  return ''

def getHeader(section) -> str:
  headerpath = section.xpath('header')  
  if len(headerpath) > 0:
    return headerpath[0].text
  return ''


def text_to_vect(txt: str , ngram_size: int = 4):
    """
    Gets ngrams from text
    """
    # See https://stackoverflow.com/a/32128803/628748
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(txt)
    #vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,ngram_size),
    #    tokenizer=TreebankWordTokenizer().tokenize, lowercase=True)
    vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,ngram_size),
        tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    vect.fit(sentences)
    # ngrams = vect.get_feature_names_out()
    # print('{1}-grams: {0}'.format(ngrams, ngram_size))
    #print(vect.vocabulary_)
    return vect # list of text documents

def xml_to_sections(xml_path: str):
    """
    Parses the xml file into sections 
    """
    try:
        billTree = etree.parse(xml_path)
    except:
        raise Exception('Could not parse bill')
    sections = billTree.xpath('//uslm:section', namespaces=NAMESPACES)
    if len(sections) == 0:
        print('No sections found')
        return []
    return [{
            'section_number': getEnum(section) ,
            'section_header':  getHeader(section),
            'section_text': etree.tostring(section, method="text", encoding="unicode"),
            'section_xml': etree.tostring(section, method="xml", encoding="unicode")
        } if (section.xpath('header') and len(section.xpath('header')) > 0  and section.xpath('enum') and len(section.xpath('enum'))>0) else
        {
            'section_number': '',
            'section_header': '', 
            'section_text': etree.tostring(section, method="text", encoding="unicode"),
            'section_xml': etree.tostring(section, method="xml", encoding="unicode")
        } 
        for section in sections ]

def xml_to_text(xml_path: str, level: str = 'section', separator: str = '\n*****\n') -> str:
    """
    Parses the xml file and returns the text of the body element, if any
    """
    try:
        billTree = etree.parse(xml_path)
    except:
        raise Exception('Could not parse bill')
    #return etree.tostring(billTree, method="text", encoding="unicode")
    # Use 'body' for level to get the whole body element
    sections = billTree.xpath('//uslm:'+level, namespaces=NAMESPACES)
    if len(sections) == 0:
        print('No sections found')
        return '' 
    return separator.join([etree.tostring(section, method="text", encoding="unicode") for section in sections])

def xml_to_vect(xml_paths: List[str], ngram_size: int = 4):
    """
    Parses the xml file and returns the text of the body element, if any
    """
    total_str = '\n'.join([xml_to_text(xml_path) for xml_path in xml_paths])
    return text_to_vect(total_str, ngram_size=ngram_size)

    # to get the vocab dict: vect.vocabulary_

def combine_vocabs(vocabs: List[CountVectorizer]):
    """
    Combines one or more vocabs into one
    """
    vocab_keys = list(set([list(v.vocabulary_.keys()) for v in vocabs]))
    vocab = {vocab_key: str(i) for i, vocab_key in enumerate(vocab_keys)}
    return vocab

def getSampleText(level = 'body'):
    return xml_to_text(BIG_BILLS_PATHS[0])

def transform_text(text: str, vocab: dict, ngram_size: int = 4):
    """
    Transforms text into a vector using the vocab
    """
    return CountVectorizer(vocabulary=vocab).fit_transform([text])

def train_count_vectorizer(train_data: List[str], ngram_size: int = 4):
    """
    Trains a count vectorizer on the training data
    """
    vectorizer = CountVectorizer(ngram_range=(ngram_size,ngram_size), preprocessor=xml_to_text, tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    with ExitStack() as stack:
        files = [
            stack.enter_context(open(filename))
            for filename in train_data 
        ]
        X = vectorizer.fit_transform(files)
    return vectorizer, X 

def train_hashing_vectorizer(train_data: List[str], ngram_size: int = 4):
    """
    Trains a hashing vectorizer on the training data
    """
    vectorizer = HashingVectorizer(ngram_range=(ngram_size,ngram_size), preprocessor=xml_to_text, tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    with ExitStack() as stack:
        files = [
            stack.enter_context(open(filename))
            for filename in train_data 
        ]
        X = vectorizer.fit_transform(files)
    return vectorizer, X

def test_hashing_vectorizer(vectorizer: HashingVectorizer, test_data: List[str]):
    return vectorizer.transform(test_data)

def xml_samples_to_text(dirpath: str, level: str = 'section', separator: str = '\n*****\n'):
    """
    Converts xml files in a directory to txt files
    """
    xfiles = get_filepaths(dirpath)
    for xfile in xfiles:
        with open(xfile.replace('.xml', f'-{level}s.txt'), 'w') as f:
            f.write(xml_to_text(xfile, level=level, separator=separator))

# TODO: Add a function to parse the bill (text) into paragraphs 

# TODO: create a streaming hash vectorizer. See 
# https://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html#sphx-glr-auto-examples-applications-plot-out-of-core-classification-py