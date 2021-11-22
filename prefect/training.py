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
from contextlib import ExitStack
from lxml import etree 
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer, TreebankWordTokenizer
from scipy.spatial.distance import cosine
import prefect
from prefect import task, Flow, Parameter
from prefect.executors import LocalDaskExecutor



# Among the larger bills is samples/congress/116/BILLS-116s1790enr.xml (~ 10MB)

PATH_116_USLM = '../samples/congress/116/uslm'
PATH_117_USLM = '../samples/congress/117/uslm'


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


# Accepts array of directories containing USLM bills (116th, congress, 117th congress etc) 
# Returns file paths 
@task(log_stdout=True)
def get_bill_file_paths(dirs):
    bill_files = []
    print('Finding bill files')

    for d in dirs:
        if os.path.isdir(d) == False:
            print("Bill directory not found")

        for f in os.listdir(d):
            if f.endswith('.xml'):
                print(f)
                bill_files.append(os.path.join(d, f))
    
    print(f'{len(bill_files)} bill files found')
    return bill_files

# Accepts list of fully qualified USLM bill paths
# Returns 
@task(log_stdout=True)
def extract_transform_load_bills(bill_files): 
    print('Beginning ETL step')

    doc_corpus_data=[]
    section_corpus_data = []

    for i in range(0, len(bill_files)):
        bill_doc_file = bill_files[i]
        #parse xml into sections
        secs = xml_to_sections(bill_doc_file)
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
        else:
            print("No sections found")

    print(doc_corpus_data[4])
    print(section_corpus_data[4])

    #get only whole document content from doc_corpus_data list
    only_doc_data = [row[1] for row in doc_corpus_data]
    #get only section content from section_corpus_data list
    only_section_data = [row[2] for row in section_corpus_data]

    #store pre-processed document corpus and section level corpus
    # todo: index bill text in elasticsearch instead? 
    pickle.dump(doc_corpus_data, open("tv_doc_corpus_data.pickel", "wb"))
    pickle.dump(section_corpus_data, open("tv_section_corpus_data.pickel", "wb"))
    #get length of only_doc_data list
    print(f'{len(only_doc_data)} documents found in corpus')
    print(f'{len(only_section_data)} sections found in corpus')

    return only_doc_data, only_section_data
   
@task(log_stdout=True)
def vectorize_corpus(corpus_data, output_filename_prefix):
    print(f'Creating {output_filename_prefix} vectorization')

    sec_tfidf_vectorizer = TfidfVectorizer(ngram_range=(4,4), tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    #Fit tfidf vectorize instance on section level corpus
    tv_section_matrix = sec_tfidf_vectorizer.fit_transform(corpus_data)

    # save tfidf vectorize instance for only_doc_data
    pickle.dump(tfidf_vectorizer, open(f'{output_filename_prefix}_tfidf_vectorizer.pickel', "wb"))
    # load tfidf vectorize instance for only_doc_data
    tfidf_vectorizer = pickle.load(open(f'{output_filename_prefix}_tfidf_vectorizer.pickel', "rb"))

    return tfidf_vectorizer


with Flow("Training", executor=LocalDaskExecutor()) as flow:
    file_paths = get_bill_file_paths([PATH_117_USLM, PATH_116_USLM])
    bill_data = extract_transform_load_bills(file_paths)
    vectorize_corpus(bill_data[0], "document")
    vectorize_corpus(bill_data[1], "section")

flow.register(project_name="BillSimilarityEngine")