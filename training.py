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
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Among the larger bills is samples/congress/116/BILLS-116s1790enr.xml (~ 10MB)

PATH_NEW = './samples/congress/117/new'
PATH_116_USLM = './samples/congress/116/uslm'
PATH_117_USLM = './samples/congress/117/uslm'
PATH_116_USLM_TRAIN = './samples/congress/116/train'
PATH_116_TEXT = './samples/congress/116/txt'


BILLS_SAMPLE = [f'BILLS-116hr{number}ih.xml' for number in range(100, 300)]
BIG_BILLS = ['BILLS-116s1790enr.xml', 'BILLS-116hjres31enr.xml']
BIG_BILLS_PATHS = [path.join(PATH_116_USLM, bill) for bill in (BIG_BILLS + BILLS_SAMPLE)]

SAMPLE_BILL_PATHS_TRAIN = [join(PATH_116_USLM_TRAIN, f) for f in listdir(PATH_116_USLM) if isfile(join(PATH_116_USLM_TRAIN, f))]
SAMPLE_BILL_PATHS = [join(PATH_117_USLM, f) for f in listdir(PATH_117_USLM) if isfile(join(PATH_117_USLM, f))]


NAMESPACES = {'uslm': 'http://xml.house.gov/schemas/uslm/1.0'}


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

#record training time for both vectorizer
start = time.time()


doc_corpus_data=[]
section_corpus_data = []

#get all xml files from data directory for parsing
bill_files = [f for f in os.listdir(PATH_117_USLM) if f.endswith('.xml')]

#iterate over all bill files
for i in range(0, len(bill_files)):
    
    #indexing bill document file
    bill_doc_file = bill_files[i]
    
    #parse xml into sections
    secs = xml_to_sections(os.path.join(PATH_117_USLM, bill_doc_file))
    
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
    

#get only whole document content from doc_corpus_data list
only_doc_data = [row[1] for row in doc_corpus_data]

#get only section content from section_corpus_data list
only_section_data = [row[2] for row in section_corpus_data]

#store pre-processed document corpus and section level corpus
pickle.dump(doc_corpus_data, open("tv_doc_corpus_data.pickel", "wb"))
pickle.dump(section_corpus_data, open("tv_section_corpus_data.pickel", "wb"))

#get length of only_doc_data list
print(len(only_doc_data))

#get length of only_section_data list
print(len(only_section_data))

done = time.time()
elapsed = done - start
print('Time took in ETL with {} xml data files is {}'.format(len(only_doc_data), elapsed))   

#record training time for both vectorizer
start = time.time()


# Vectorizer to convert a collection of raw documents to a matrix 
doc_tfidf_vectorizer = TfidfVectorizer(ngram_range=(4,4), tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
#Fit tfidf vectorize instance on document level corpus
tv_doc_matrix = doc_tfidf_vectorizer.fit_transform(only_doc_data)

# Vectorizer to convert a collection of sections to a matrix 
sec_tfidf_vectorizer = TfidfVectorizer(ngram_range=(4,4), tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
#Fit tfidf vectorize instance on section level corpus
tv_section_matrix = sec_tfidf_vectorizer.fit_transform(only_section_data)


done = time.time()
elapsed = done - start
print("Time took in training of both vectorizer(s) ", elapsed)

# save tfidf vectorize instance for only_doc_data
pickle.dump(doc_tfidf_vectorizer, open("doc_tfidf_vectorizer.pickel", "wb"))
# load tfidf vectorize instance for only_doc_data
doc_tfidf_vectorizer = pickle.load(open("doc_tfidf_vectorizer.pickel", "rb"))

#save tfidf vectorize instance for only_section_data
pickle.dump(sec_tfidf_vectorizer, open("sec_tfidf_vectorizer.pickel", "wb"))
# load tfidf vectorize instance for only_section_data
sec_tfidf_vectorizer = pickle.load(open("sec_tfidf_vectorizer.pickel", "rb"))