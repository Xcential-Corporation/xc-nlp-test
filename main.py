# python 
import json
import pandas as pd
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

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Among the larger bills is samples/congress/116/BILLS-116s1790enr.xml (~ 10MB)

PATH_116_USLM = 'data/samples/congress/116/uslm'
PATH_117_USLM = 'data/samples/congress/117/uslm'
PATH_116_USLM_TRAIN = 'data/samples/congress/116/train'
PATH_116_TEXT = 'data/samples/congress/116/txt'

BILLS_SAMPLE = [f'BILLS-116hr{number}ih.xml' for number in range(100, 300)]
BIG_BILLS = ['BILLS-116s1790enr.xml', 'BILLS-116hjres31enr.xml']
BIG_BILLS_PATHS = [path.join(PATH_116_USLM, bill) for bill in (BIG_BILLS + BILLS_SAMPLE)]

SAMPLE_BILL_PATHS_TRAIN = [join(PATH_116_USLM_TRAIN, f) for f in listdir(PATH_116_USLM) if isfile(join(PATH_116_USLM_TRAIN, f))]
SAMPLE_BILL_PATHS = [join(PATH_117_USLM, f) for f in listdir(PATH_117_USLM) if isfile(join(PATH_117_USLM, f))]


NAMESPACES = {'uslm': 'http://xml.house.gov/schemas/uslm/1.0'}


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
    #vect = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(ngram_size,ngram_size),
    #    tokenizer=TreebankWordTokenizer().tokenize, lowercase=True)
    vect = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(ngram_size,ngram_size),
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

def combine_vocabs(vocabs: List[TfidfVectorizer]):
    """
    Combines one or more vocabs into one
    """
    vocab_keys = list(set([list(v.vocabulary_.keys()) for v in vocabs]))
    vocab = {vocab_key: str(i) for i, vocab_key in enumerate(vocab_keys)}
    return vocab

def get_combined_vocabs(xml_paths: List[str] = SAMPLE_BILL_PATHS, ngram_size: int = 4):
    """
    Gets the combined vocabulary of all the xml files
    """
    return xml_to_vect(xml_paths, ngram_size=ngram_size)

def getSampleText(level = 'body'):
    return xml_to_text(BIG_BILLS_PATHS[0])

def transform_text(text: str, vocab: dict, ngram_size: int = 4):
    """
    Transforms text into a vector using the vocab
    """
    return TfidfVectorizer(vocabulary=vocab).fit_transform([text])

def train_Tfidf_vectorizer(train_data: List[str], ngram_size: int = 4):
    """
    Trains a Tfidf vectorizer on the training data
    """
    vectorizer = TfidfVectorizer(ngram_range=(ngram_size,ngram_size), preprocessor=xml_to_text, tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    with ExitStack() as stack:
        files = [
            stack.enter_context(open(filename))
            for filename in train_data 
        ]
        X = vectorizer.fit_transform(files)
    return vectorizer, X 


def xml_samples_to_text(dirpath: str, level: str = 'section', separator: str = '\n*****\n'):
    """
    Converts xml files in a directory to txt files
    """
    xfiles = get_filepaths(dirpath)
    for xfile in xfiles:
        with open(xfile.replace('.xml', f'-{level}s.txt'), 'w') as f:
            f.write(xml_to_text(xfile, level=level, separator=separator))




#clean text 
def text_cleaning(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text



"""
create document and sections data list after xml document level and section level parsing

"""

def create_document_and_sections_data_list(directory_path):
    
    #create data lists 
    doc_corpus_data=[]
    section_corpus_data = []

    #get all xml files from data directory for parsing
    bill_files = [f for f in os.listdir(directory_path) if f.endswith('.xml')]

    print(len(bill_files))
    #bill_files = bill_files[:10]

    #iterate over all bill files
    for i in range(0, len(bill_files)):

        #indexing bill document file
        bill_doc_file = bill_files[i]

        #parse xml into sections
        secs = xml_to_sections(os.path.join(directory_path, bill_doc_file))

        #check  of sections should be 1 or more than 1
        if(len(secs)>0):  

            #intialize string variable for document content
            doc_content = ""

            #iterate over all parse sections text of bill doc file
            for s_number, section in enumerate(secs):  

                #text cleaning applied on each section text
                sec_text = text_cleaning(section['section_text'])

                #concatenate section text to doc content 
                doc_content = doc_content + sec_text + " "

                 #for now sentence id is sentence number in document
                section_corpus_data.append([Path(bill_doc_file).stem[:], s_number, sec_text ])

            doc_corpus_data.append([Path(bill_doc_file).stem[:], doc_content])


    return doc_corpus_data, section_corpus_data


"""
create pandas_dataframe for storing bill document and sections data with their ID and Text

"""

def create_pandas_dataframe_for_bill_document_and_sections(doc_corpus_data, section_corpus_data):
    
    
    # doc_corpus_columns
    doc_corpus_columns = ["Bill ID", 'Document Text']
    doc_corpus_data_df = pd.DataFrame(doc_corpus_data, columns = doc_corpus_columns)
    
    #section_corpus_columns
    section_corpus_columns = ["Bill ID", 'Section Number', 'Section Text']
    section_corpus_data_df = pd.DataFrame(section_corpus_data, columns = section_corpus_columns)
    
    return doc_corpus_data_df, section_corpus_data_df


"""
Save Vocabulary Corpus of Bill Documents and Section to Parquet File
"""

def save_bill_documents_and_section_corpuses_to_parquet_file(doc_corpus_data_df, section_corpus_data_df):

    doc_corpus_data_df.to_parquet('doc_corpus_data.parquet', engine='fastparquet')
    section_corpus_data_df.to_parquet('section_corpus_data.parquet', engine='fastparquet')
    

"""
Load Vocabulary Corpus of Bill Documents and Section From Parquet File
"""
def load_bill_documents_and_section_corpuses_from_parquet_file():

    doc_corpus_data_df = pd.read_parquet('doc_corpus_data.parquet', engine='fastparquet')
    section_corpus_data_df = pd.read_parquet('section_corpus_data.parquet', engine='fastparquet')
    
    return doc_corpus_data_df, section_corpus_data_df


"""
Train TfIDF Vectorizer for Bill Documents and Sections using Pandas Dataframe
"""

def train_tfidf_vectorizer_on_bill_document_and_sections(doc_corpus_data_df, section_corpus_data_df):

    # Vectorizer to convert a collection of raw documents to a matrix 
    doc_tfidf_vectorizer = TfidfVectorizer(ngram_range=(4,4), tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    #Fit tfidf vectorize instance on document level corpus
    tv_doc_matrix = doc_tfidf_vectorizer.fit_transform(doc_corpus_data_df['Document Text'])

    # Vectorizer to convert a collection of sections to a matrix 
    sec_tfidf_vectorizer = TfidfVectorizer(ngram_range=(4,4), tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    #Fit tfidf vectorize instance on section level corpus
    tv_section_matrix = sec_tfidf_vectorizer.fit_transform(section_corpus_data_df['Section Text'])
    
    return doc_tfidf_vectorizer, sec_tfidf_vectorizer


"""
Store TfIDF Vectorizer Model for Bill Documents and Sections
"""

def store_bill_document_and_section_tfidf_vectorizer(doc_tfidf_vectorizer, sec_tfidf_vectorizer):
    
    # save tfidf vectorize instance for doc_tfidf_vectorizer
    pickle.dump(doc_tfidf_vectorizer, open("doc_tfidf_vectorizer.pickel", "wb"))

    #save tfidf vectorize instance for sec_tfidf_vectorizer
    pickle.dump(sec_tfidf_vectorizer, open("sec_tfidf_vectorizer.pickel", "wb"))


    
"""
Load TfIDF Vectorizer Model for Bill Documents and Sections
"""    
def laod_bill_document_and_section_tfidf_vectorizer():
    
    # load tfidf vectorize instance for doc_tfidf_vectorizer
    doc_tfidf_vectorizer = pickle.load(open("doc_tfidf_vectorizer.pickel", "rb"))


    # load tfidf vectorize instance for sec_tfidf_vectorizer
    sec_tfidf_vectorizer = pickle.load(open("sec_tfidf_vectorizer.pickel", "rb"))
    
    return doc_tfidf_vectorizer, sec_tfidf_vectorizer



if __name__ == '__main__':

    
    #create document and sections data list after xml document level and section level parsing
    
    directory_path = PATH_117_USLM
    doc_corpus_data, section_corpus_data = create_document_and_sections_data_list(directory_path)
    
    
    #create pandas_dataframe for storing bill document and sections data with their ID and Text
    doc_corpus_data_df, section_corpus_data_df = create_pandas_dataframe_for_bill_document_and_sections(doc_corpus_data, section_corpus_data)
    
    
    #Save Vocabulary Corpus of Bill Documents and Section to Parquet File
    save_bill_documents_and_section_corpuses_to_parquet_file(doc_corpus_data_df, section_corpus_data_df)
    
    #Train TfIDF Vectorizer for Bill Documents and Sections using Pandas Dataframe
    doc_tfidf_vectorizer, sec_tfidf_vectorizer = train_tfidf_vectorizer_on_bill_document_and_sections(doc_corpus_data_df, section_corpus_data_df)
    store_bill_document_and_section_tfidf_vectorizer(doc_tfidf_vectorizer, sec_tfidf_vectorizer)
    
    
    
