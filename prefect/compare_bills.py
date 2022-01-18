# flow that parallelizes bill-level comparisons

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

import re

import billsim
from billsim import utils_db, pymodels


def get_bills_to_compare()
    A_doc_name = '117hr1319enr'
    B_doc_name = '117hr1319eh'

    return [[A_doc_name, B_doc_name]]


def calculate_bill_similarity(bill_pair):
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

    bill_a = bill_pair[0]
    bill_b = bill_pair[1]
    #pick any Document A & any Document B from data lists (at least that have more than 1 section)
    A_doc_name = "BILLS-" + bill_a
    B_doc_name = "BILLS-" + bill_b
 
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

    # save to db
    ids = utils_db.get_bill_ids(bill_pair)

    if len(ids) == 2 and ids[bill_a] != ids[bill_b]:
        print("Both bills found. Saving.")
        bill_id = ids[bill_a]
        bill_to_id = ids[bill_b]

        bill_to_bill_new = pymodels.BillToBillModel(
            billnumber_version= bill_a,
            billnumber_version_to= bill_b,
            score= from_doc_similarity[0][0],
            score_to= to_doc_similarity[0][0],
            bill_id= bill_id,
            bill_to_id= bill_to_id
        )

        btb = utils_db.save_bill_to_bill(bill_to_bill_model= bill_to_bill_new)

with Flow("Bill Level Comparisons", executor=LocalDaskExecutor()) as flow:
    bills_to_compare = get_bills_to_compare()
    calculate_bill_similarity(bills_to_compare)
    
flow.register(project_name="BillSimilarityEngine")