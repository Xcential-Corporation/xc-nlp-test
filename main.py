# python
import json
import pandas as pd
import time
import pickle
import re, string
import os
from os import path, walk, listdir
from pathlib import Path
from os.path import isfile, join
from types import new_class
from typing import List
from lxml import etree 
from contextlib import ExitStack
import sklearn.feature_extraction.text
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer, TreebankWordTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Among the larger bills is samples/congress/116/BILLS-116s1790enr.xml (~ 10MB)

PATH_116_USLM = 'samples/congress/116/uslm'
PATH_117_USLM = 'samples/congress/117'
PATH_116_USLM_TRAIN = 'samples/congress/116/train'
PATH_116_TEXT = 'samples/congress/116/txt'

BILLS_SAMPLE = [f'BILLS-116hr{number}ih.xml' for number in range(100, 300)]
BIG_BILLS = ['BILLS-116s1790enr.xml', 'BILLS-116hjres31enr.xml']
BIG_BILLS_PATHS = [path.join(PATH_116_USLM, bill) for bill in (BIG_BILLS + BILLS_SAMPLE)]

SAMPLE_BILL_PATHS_TRAIN = [join(PATH_116_USLM_TRAIN, f) for f in listdir(PATH_116_USLM) if isfile(join(PATH_116_USLM_TRAIN, f))]
SAMPLE_BILL_PATHS = [join(PATH_117_USLM, f) for f in listdir(PATH_117_USLM) if isfile(join(PATH_117_USLM, f))]


NAMESPACES = {'uslm': 'http://xml.house.gov/schemas/uslm/1.0'}


def get_filepaths(dirpath: str, reMatch = r'.xml$') -> List[str]:
    filepaths = []
    for (dirpath, dirnames, filenames) in walk(dirpath):
        for filename in filenames:
            if re.search(reMatch, filename):
                filepaths.append(path.join(dirpath, filename))
    return filepaths 

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
    #print("list of text documents: ", vect)
    return vect # list of text documents

def xml_to_sections(xml_path: str):
    #print(xml_path)
    
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
    return CountVectorizer(vocabulary=vocab).fit_transform([text])

def train_Count_vectorizer(train_data: List[str], ngram_size: int = 4):
    """
    Trains a Count vectorizer on the training data
    """
    vectorizer = CountVectorizer(ngram_range=(ngram_size,ngram_size), preprocessor=xml_to_text, tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
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
    bill_files = bill_files[:10]
    
    
    

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
                
                #print(text_to_vect(sec_text, ngram_size=4))

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
Save documents dataframe of Bill Documents and Section to Parquet File
"""

def save_bill_documents_and_section_corpuses_to_parquet_file(doc_corpus_data_df, section_corpus_data_df):

    doc_corpus_data_df.to_parquet('doc_corpus_data_df.parquet', engine='fastparquet')
    section_corpus_data_df.to_parquet('section_corpus_data_df.parquet', engine='fastparquet')
    

"""
Load documents dataframe of Bill Documents and Section From Parquet File
"""
def load_bill_documents_and_section_corpuses_from_parquet_file():

    doc_corpus_data_df = pd.read_parquet('doc_corpus_data_df.parquet', engine='fastparquet')
    section_corpus_data_df = pd.read_parquet('section_corpus_data_df.parquet', engine='fastparquet')
    
    return doc_corpus_data_df, section_corpus_data_df



"""
Save Vocabulary of Bill Documents and Section to Parquet File
"""

def save_bill_documents_and_section_vocab_to_parquet_file(doc_vocab, section_vocab):
    
    doc_vocab = pd.DataFrame([doc_vocab])
    section_vocab = pd.DataFrame([section_vocab])

    doc_vocab.to_parquet('doc_vocab.parquet', engine='fastparquet')
    section_vocab.to_parquet('section_vocab.parquet', engine='fastparquet')
    

"""
Load Vocabulary of Bill Documents and Section From Parquet File
"""
def load_bill_documents_and_section_vocab_from_parquet_file():

    doc_vocab = pd.read_parquet('doc_vocab.parquet', engine='fastparquet')
    section_vocab = pd.read_parquet('section_vocab.parquet', engine='fastparquet')
    
    return doc_vocab, section_vocab




"""
Train Count Vectorizer for Bill Documents and Sections using Pandas Dataframe
"""

def train_count_vectorizer_on_bill_document_and_sections(doc_corpus_data_df, section_corpus_data_df):

    # Vectorizer to convert a collection of raw documents to a matrix 
    doc_count_vectorizer = CountVectorizer(ngram_range=(4,4), tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    #Fit count vectorize instance on document level corpus
    cv_doc_matrix = doc_count_vectorizer.fit_transform(doc_corpus_data_df['Document Text'])
    
    
    # Vectorizer to convert a collection of sections to a matrix 
    sec_count_vectorizer = CountVectorizer(ngram_range=(4,4), tokenizer=RegexpTokenizer(r"\w+").tokenize, lowercase=True)
    #Fit count vectorize instance on section level corpus
    cv_section_matrix = sec_count_vectorizer.fit_transform(section_corpus_data_df['Section Text'])
    
    #print("CV section: ", cv_section_matrix.todense().tolist())
    
    #save vocabulary
    save_bill_documents_and_section_vocab_to_parquet_file(doc_count_vectorizer.vocabulary_, sec_count_vectorizer.vocabulary_)
  
    return doc_count_vectorizer, sec_count_vectorizer


"""
Store Count Vectorizer Model for Bill Documents and Sections
"""

def store_bill_document_and_section_count_vectorizer(doc_count_vectorizer, sec_count_vectorizer):
    
    # save count vectorize instance for doc_count_vectorizer
    pickle.dump(doc_count_vectorizer, open("doc_count_vectorizer.pickel", "wb"))

    #save count vectorize instance for sec_count_vectorizer
    pickle.dump(sec_count_vectorizer, open("sec_count_vectorizer.pickel", "wb"))


    
"""
Load Count Vectorizer Model for Bill Documents and Sections
"""    
def laod_bill_document_and_section_count_vectorizer():
    
    # load count vectorize instance for doc_count_vectorizer
    doc_count_vectorizer = pickle.load(open("doc_count_vectorizer.pickel", "rb"))


    # load count vectorize instance for sec_count_vectorizer
    sec_count_vectorizer = pickle.load(open("sec_count_vectorizer.pickel", "rb"))
    
    return doc_count_vectorizer, sec_count_vectorizer


def create_vectorized_dataframe_from_all_documents(doc_corpus_data_df, section_corpus_data_df):
    
    doc_count_vectorizer, sec_count_vectorizer  = laod_bill_document_and_section_count_vectorizer()

    vectorized_doc_corpus_data = []
    for doc in range(len(doc_corpus_data_df)):

        bill_doc_list = doc_corpus_data_df.iloc[doc].values
        bill_id = bill_doc_list[0]
        bill_document_text = bill_doc_list[1]

        bill_doc_vectorized = doc_count_vectorizer.transform([bill_document_text])
        

        vectorized_doc_corpus_data.append([bill_id, {"doc": bill_doc_vectorized.todense().tolist()}])


    vectorized_section_corpus_data = []
    for section in range(len(section_corpus_data_df)):

        bill_doc_list = section_corpus_data_df.iloc[section].values

        bill_id = bill_doc_list[0]
        bill_section_id = bill_doc_list[1]
        bill_section_text = bill_doc_list[2]

        bill_sec_vectorized = sec_count_vectorizer.transform([bill_section_text])
        
        
        vectorized_section_corpus_data.append([bill_id, bill_section_id+1, {"doc": bill_sec_vectorized.todense().tolist()}])
        
        
        
    # vectorized_doc_df
    vectorized_doc_columns = ["Bill ID", 'Vectorized Document']
    
    


    vectorized_doc_df = pd.DataFrame(vectorized_doc_corpus_data, columns = vectorized_doc_columns)
    
    

    vectorized_doc_df['Vectorized Document'] = vectorized_doc_df['Vectorized Document'].astype('object') 
    
    #vectorized_section_df
    vectorized_section_columns = ["Bill ID", 'Section Number', 'Vectorized Section']
    
    
    vectorized_section_df = pd.DataFrame(vectorized_section_corpus_data, columns = vectorized_section_columns)
    
    vectorized_section_df['Vectorized Section'] = vectorized_section_df['Vectorized Section'].astype('object') 


    
    
    
    return vectorized_doc_df, vectorized_section_df

    
"""
Save vectorized documents dataframe of Bill Documents and Section to Parquet File
"""

def save_bill_documents_and_section_corpuses_to_parquet_file(vectorized_doc_df, vectorized_section_df):

    vectorized_doc_df.to_parquet('vectorized_doc_df.parquet', engine='fastparquet')
    vectorized_section_df.to_parquet('vectorized_section_df.parquet', engine='fastparquet')
    

"""
Load vectorized documents dataframe of Bill Documents and Section From Parquet File
"""
def load_bill_documents_and_section_corpuses_from_parquet_file():

    vectorized_doc_df = pd.read_parquet('vectorized_doc_df.parquet', engine='fastparquet')
    vectorized_section_df = pd.read_parquet('vectorized_section_df.parquet', engine='fastparquet')
    
    return vectorized_doc_df, vectorized_section_df



#transform document into vectorized space
def document_count_vectorized_transformation(document, doc_count_vectorizer):
    
    doc_vectorized = doc_count_vectorizer.transform([document])
    return doc_vectorized

def section_doc_count_vectorized_transformation(section_doc, sec_count_vectorizer):
    
    section_doc_vectorized = sec_count_vectorizer.transform(section_doc)
    return section_doc_vectorized

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

    




def calculate_document_similarity_from_pandas_dataframe(bill_id):

    temp = vectorized_doc_df.loc[vectorized_doc_df['Bill ID'] == bill_id]

    df = vectorized_doc_df

    result_doc = []

    #find_similar_df = df[df["Bill ID"] == bill_id]
    whole_df = df[df["Bill ID"] != bill_id]

    for i in range(len(whole_df)):



        time_,sim_score = cosine_pairwise_sim(temp.iloc[0]['Vectorized Document']['doc'], 
                                              whole_df.iloc[i]['Vectorized Document']['doc'])



        result_doc.append([bill_id , whole_df.iloc[i]['Bill ID'],sim_score[0][0]])


    result_doc_df_columns = ["Source Bill ID", 'Target Bill ID', 'Doc Sim Score']


    result_doc_df = pd.DataFrame(result_doc, columns = result_doc_df_columns)
    return result_doc_df








def calculate_section_similarity_from_pandas_dataframe(bill_id):

    temp_1 = vectorized_section_df[vectorized_section_df["Bill ID"] != bill_id]
    temp_2 = vectorized_section_df.loc[vectorized_section_df['Bill ID'] == bill_id]

    result_doc =[]

    for i in range(len(temp_1)):
        #print(temp_1.iloc[i])
        #print(temp.iloc[i]['Bill ID'])
        #print(temp.iloc[i]['Vectorized Section'])

        for j in range(len(temp_2)):


            time_,sim_score = cosine_pairwise_sim(temp_1.iloc[i]['Vectorized Section']['doc'], 
                                              temp_2.iloc[j]['Vectorized Section']['doc'])

            result_doc.append([ temp_2.iloc[j]['Bill ID'], 
                  temp_2.iloc[j]['Section Number'],temp_1.iloc[i]['Bill ID'], temp_1.iloc[i]['Section Number'],

                  sim_score[0][0]])

            #print(temp_2.iloc[j]['Bill ID'], 
                  #temp_2.iloc[j]['Section Number'], temp_1.iloc[i]['Bill ID'], temp_1.iloc[i]['Section Number'], 

                  #sim_score)



    result_sec_df_columns = ["Source Bill ID", "Source Section Number", 'Target Bill ID', 'Target Section Number', 'Section Sim Score']


    result_sec_df = pd.DataFrame(result_doc, columns = result_sec_df_columns)
    return result_sec_df       

    
    
def calculate_similarity_by_matrix_multiplication_from_pandas_dataframe(bill_id):
    
    result_doc_df = calculate_document_similarity_from_pandas_dataframe(bill_id)
    result_sec_df = calculate_section_similarity_from_pandas_dataframe(bill_id)
    
    merged_data= result_sec_df.merge(result_doc_df, on=["Source Bill ID","Target Bill ID"])
   
    
    return merged_data   
    


def filters_the_results_by_two_thresholds(result_df, threshold_for_whole_bill, threshold_for_sections):
    
    result_df = result_df.loc[(result_df['Doc Sim Score'] >= threshold_for_whole_bill) & (result_df['Section Sim Score'] >= threshold_for_sections)]
    
    return result_df




if __name__ == '__main__':
    
    #threshold for whole bill  
    threshold_for_whole_bill = 0.7
    
    #threshold for sections.
    threshold_for_sections = 0.3

    
    #create document and sections data list after xml document level and section level parsing
    
    directory_path = PATH_117_USLM
    doc_corpus_data, section_corpus_data = create_document_and_sections_data_list(directory_path)
    
    
    #create pandas_dataframe for storing bill document and sections data with their ID and Text
    doc_corpus_data_df, section_corpus_data_df = create_pandas_dataframe_for_bill_document_and_sections(doc_corpus_data, section_corpus_data)
    
    
    #Save Vocabulary Corpus of Bill Documents and Section to Parquet File
    save_bill_documents_and_section_corpuses_to_parquet_file(doc_corpus_data_df, section_corpus_data_df)
    
    #Train Count Vectorizer for Bill Documents and Sections using Pandas Dataframe
    doc_count_vectorizer, sec_count_vectorizer = train_count_vectorizer_on_bill_document_and_sections(doc_corpus_data_df, section_corpus_data_df)
    store_bill_document_and_section_count_vectorizer(doc_count_vectorizer, sec_count_vectorizer)
    
    
    #create vectorized dataframe
    vectorized_doc_df, vectorized_section_df = create_vectorized_dataframe_from_all_documents(doc_corpus_data_df, section_corpus_data_df)
    
    
    #save vectorized dfs into parquet format
    save_bill_documents_and_section_corpuses_to_parquet_file(vectorized_doc_df, vectorized_section_df)
    
    #load vectorized dfs from memory
    vectorized_doc_df, vectorized_section_df = load_bill_documents_and_section_corpuses_from_parquet_file()
    
    
    
    

    #Enter any Bill ID to find similarity using matrix multiplication of [bill1]*[ALL BILLS] to calculate all similiarities at once.
    bill_id = "BILLS-117hconres11eh"

    result_df = calculate_similarity_by_matrix_multiplication_from_pandas_dataframe(bill_id)

    #Filters the results by two thresholds: a) threshold for whole bill and b) threshold for sections.
    result_df  = filters_the_results_by_two_thresholds(result_df, threshold_for_whole_bill, threshold_for_sections)
    
    print(result_df)











