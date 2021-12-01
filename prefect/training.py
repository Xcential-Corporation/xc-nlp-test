# python
import json
import time
import pickle
import re, string
import os
from os import path, listdir
from os.path import isfile, join
from pathlib import Path
from lxml import etree 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
import prefect
from prefect import task, Flow, Parameter
from prefect.executors import LocalDaskExecutor

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

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(4,4), tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)

    tv_section_matrix = tfidf_vectorizer.fit_transform(corpus_data)

    pickle.dump(tfidf_vectorizer, open(f'{output_filename_prefix}_tfidf_vectorizer.pickel', "wb"))

    return tfidf_vectorizer

# compute cosine pairwise similarity
def cosine_pairwise_sim(a_vectorized, b_vectorized):
    
    #record time for computing similarity 
    start = time.time()

    sim_score =  cosine_similarity(a_vectorized, b_vectorized)

    done = time.time()
    elapsed = done - start
    return elapsed, sim_score


#create list response
def create_list_response(A_doc_name, B_doc_name, doc_sim_score, sec_doc_sim_score):
    
    #record time for creating list response
    start = time.time()
    
    #create result list
    res_list = []

    #create empty list
    temp=[]
    temp.append("ORIGINAL DOCUMENT ID: " + A_doc_name)
    temp.append("MATCHED DOCUMENT ID: " + B_doc_name)
    temp.append("DOCUMENT SIMILARITY SCORE: " + str(doc_sim_score[0][0]))

    #iterate over sec_doc_sim_score list 
    for i, section_score_list in enumerate(sec_doc_sim_score):
        
        #add original document sentence id number
        temp.append("ORIGINAL SENTENCE ID: " + str(i+1))
           
        #sort similarity score of sections list
        section_score_list = list(enumerate(section_score_list))
        sorted_section_score_list = sorted(section_score_list, key=lambda x: x[1], reverse=True)
        
        #iterate over section level score only 
        for j, sim_score in sorted_section_score_list:
            temp.append({"MATCHED DOCUMENT ID":  B_doc_name, "MATCHED SENTENCE ID": j+1 , "SENTENCE SIMILARITY SCORE":  sim_score})

    res_list.append(temp)
    
    
    done = time.time()
    elapsed = done - start
    
    return elapsed, res_list

#transform document into vectorized space
def document_tfidf_vectorized_transformation(document, doc_tfidf_vectorizer):
    doc_vectorized = doc_tfidf_vectorizer.transform([document])
    return doc_vectorized

def section_doc_tfidf_vectorized_transformation(section_doc, sec_tfidf_vectorizer):
    section_doc_vectorized = sec_tfidf_vectorizer.transform(section_doc)
    return section_doc_vectorized


@task(log_stdout=True)
def calculate_bill_similarity(doc_corpus_data, doc_tfidf_vectorizer, section_corpus_data, sec_tfidf_vectorizer):
    doc_corpus_data = open("tv_doc_corpus_data.pickel", "rb")
    doc_tfidf_vectorizer = open("document_tfidf_vectorizer.pickel", "rb")
    section_corpus_data = open("tv_section_corpus_data.pickel", "rb")
    sec_tfidf_vectorizer = open("section_tfidf_vectorizer.pickel", "rb")

    section_corpus_data = pickle.load(section_corpus_data)
    doc_corpus_data = pickle.load(doc_corpus_data)
    doc_tfidf_vectorizer = pickle.load(doc_tfidf_vectorizer)
    sec_tfidf_vectorizer = pickle.load(sec_tfidf_vectorizer)
    
    logger = prefect.context.get("logger")

    logger.info("Starting calc bill similarity.")
    logger.warning("A warning message.")
    print('Calculating bill similarity')

    #pick any Document A & any Document B from data lists (at least that have more than 1 section)
    A_doc_name = 'BILLS-117hr1319enr'
    B_doc_name = 'BILLS-117hr1319eh'

    A_doc = [i[1] for i in doc_corpus_data if A_doc_name ==i[0]][0]
    B_doc = [i[1] for i in doc_corpus_data if B_doc_name ==i[0]][0]

    A_section_doc = [i[2] for i in section_corpus_data if A_doc_name ==i[0]]
    B_section_doc = [i[2] for i in section_corpus_data if B_doc_name ==i[0]]

    #transform document A content and document B content
    A_doc_vectorized = document_tfidf_vectorized_transformation(A_doc, doc_tfidf_vectorizer)
    B_doc_vectorized = document_tfidf_vectorized_transformation(B_doc, doc_tfidf_vectorizer)
    #transform document A section content and  document B section content
    A_section_doc_vectorized = section_doc_tfidf_vectorized_transformation(A_section_doc, sec_tfidf_vectorizer)
    B_section_doc_vectorized = section_doc_tfidf_vectorized_transformation(B_section_doc, sec_tfidf_vectorizer)

    elapsed_1, doc_sim_score = cosine_pairwise_sim(A_doc_vectorized, B_doc_vectorized)
    elapsed_2, sec_doc_sim_score = cosine_pairwise_sim(A_section_doc_vectorized, B_section_doc_vectorized)

with Flow("Training", executor=LocalDaskExecutor()) as flow:
    file_paths = get_bill_file_paths([PATH_117_USLM, PATH_116_USLM])
    bill_data = extract_transform_load_bills(file_paths)
    doc_vectors = vectorize_corpus(bill_data[0], "document")
    section_vectors = vectorize_corpus(bill_data[1], "section")
    calculate_bill_similarity(bill_data[0], doc_vectors, bill_data[1], section_vectors)
    
flow.register(project_name="BillSimilarityEngine")