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
from billsim import utils_db, pymodels
from sqlalchemy.orm import sessionmaker
from sqlmodel import create_engine


@task(log_stdout=True)

def get_bills_to_compare():
    return [["117hres158ih",	["117hres149ih"]], ["117hr1768ih",	["117s1328is"]], ["117hres318ih",	["117sres17is"]], ["117sres356is",	["117hres343ih"]], ["117s2563is",	["117hr4876ih"]], ["117s1816is",	["117hr3540ih"]], ["117hr3291ih",	["117hr3684eh"]], ["115hr4275ih",	["115hr4275rfs"]], ["117s2351is",	["117hr4459ih"]], ["117s2766is",	["117hr5466ih"]], ["116hr8939ih",	["116s160is"]], ["115hr6705ih",	["115hr3106ih"]], ["117s1588is",	["117hr263ih"]], ["117hr1992ih",	["117s799is"]], ["117s2685is",	["117hr4041ih"]], ["116hr2812ih",	["116hr2709ih"]], ["116sres178is", ["116hres319ih"]], ["110s1358is",	["110hr349ih"]], ["117hr2763ih",	["117s1322is"]]]



# def get_bills_to_compare():

#     postgres_url = f"postgresql://postgres:postgres@localhost:5433"

#     engine = create_engine(postgres_url, echo=False)

#     SessionLocal = sessionmaker(autocommit=False,
#                                 autoflush=False,
#                                 expire_on_commit=False,
#                             bind=engine)

#     query = ("SELECT billtobill.bill_id, billtobill.bill_to_id, billtobill.score_es, " 
#         "bill1.billnumber AS bill1number, bill1.version AS bill1version, "
#         "bill2.billnumber as bill2number, bill2.version AS bill2version " 
#         "FROM billtobill "
#         "LEFT JOIN bill AS bill1 ON billtobill.bill_id = bill1.id "
#         "LEFT JOIN bill AS bill2 ON billtobill.bill_to_id = bill2.id "
#         "WHERE billtobill.score_es IS NOT NULL " 
#         "ORDER BY billtobill.bill_id ASC, billtobill.score_es DESC;")

#     rs = SessionLocal().execute(query)
#     results = rs.fetchall()

#     print("Number of records with value for ES score: ", len(results))
#     # group by bill_id
#     bill_data_obj = {}
#     for result in results:
#         score_es = result[2]
#         bill1numberversion = result[3] + result[4]
#         bill2numberversion = result[5] + result[6]

#         if not bill1numberversion in bill_data_obj.keys():
#             bill_data_obj[bill1numberversion] = []

#         # Format results like so: { '116hr5001ih' : [['116hr50002ih', 631.2436789999999], ['117hr333enr', 43.4343]] }
#         bill_data_obj[bill1numberversion].append([bill2numberversion, score_es])

#     # keep only the top 30 values for es_score for each bill_id  
#     bills_to_compare = []

#     for bill_number_version, bill_list in bill_data_obj.items():
#         if len(bill_list) < 30:
#             bills_to_compare.append([ bill_number_version, bill_list ])
#         else:
#             bills_to_compare.append([bill_number_version, bill_list[0:30]])
    
#     count = 0
#     for group in bills_to_compare:
#         count = count + len(group[1])

#     print("Number of bills to compare: ", len(bills_to_compare))
#     print("Total number of comparisons to run: ", count)
    
#     return bills_to_compare



def document_tfidf_vectorized_transformation(document, doc_tfidf_vectorizer):
    doc_vectorized = doc_tfidf_vectorizer.transform([document])
    return doc_vectorized

def section_doc_tfidf_vectorized_transformation(section_doc, sec_tfidf_vectorizer):
    section_doc_vectorized = sec_tfidf_vectorizer.transform(section_doc)
    return section_doc_vectorized

# compute cosine pairwise similarity
def cosine_pairwise_sim(a_vectorized, b_vectorized):
    
    #record time for computing similarity 
    start = time.time()

    sim_score =  cosine_similarity(a_vectorized, b_vectorized)

    done = time.time()
    elapsed = done - start
    return elapsed, sim_score


@task(log_stdout=True)
def calculate_bill_similarity(bills_to_compare):

    # expects format [["117hr433ih", ["116hr435inh", "115hrenh"]],["115hr11enr", ["117hr122inh", "117hr75ih"]]]
    doc_corpus_data = open("tv_doc_corpus_data.pickel", "rb")
    doc_tfidf_vectorizer = open("document_tfidf_vectorizer.pickel", "rb")
    section_corpus_data = open("tv_section_corpus_data.pickel", "rb")
    sec_tfidf_vectorizer = open("section_tfidf_vectorizer.pickel", "rb")

    section_corpus_data = pickle.load(section_corpus_data)
    doc_corpus_data = pickle.load(doc_corpus_data)
    doc_tfidf_vectorizer = pickle.load(doc_tfidf_vectorizer)
    sec_tfidf_vectorizer = pickle.load(sec_tfidf_vectorizer)

    skip_count = 0

    for pair in bills_to_compare:
        bill_A = pair[0]
        bills_to_compare_list = pair[1]
        for bill_info in bills_to_compare_list:  
            bill_B = bill_info[0]
            print("Running comparison: ", bill_A, " ", bill_B)

            #pick any Document A & any Document B from data lists (at least that have more than 1 section)
            A_doc_name = "BILLS-" + bill_A
            B_doc_name = "BILLS-" + bill_B
        
            try:
                A_doc = [i[1] for i in doc_corpus_data if A_doc_name ==i[0]][0]
                B_doc = [i[1] for i in doc_corpus_data if B_doc_name ==i[0]][0]
            except IndexError:
                print('Could not find both bills in corpus. Skipping.')
                skip_count += 1
                continue

            #transform document A content and document B content
            A_doc_vectorized = document_tfidf_vectorized_transformation(A_doc, doc_tfidf_vectorizer)
            B_doc_vectorized = document_tfidf_vectorized_transformation(B_doc, doc_tfidf_vectorizer)

            elapsed_1, from_doc_similarity = cosine_pairwise_sim(A_doc_vectorized, B_doc_vectorized)

            # save to db
            ids = utils_db.get_bill_ids(billnumber_versions=[bill_A, bill_B], db=SessionLocal2())

            if len(ids) == 2 and ids[bill_A] != ids[bill_B]:
                print("Both bills found. Saving.")
                print("Doc similarity: ", from_doc_similarity[0][0])
                bill_id = ids[bill_A]
                bill_to_id = ids[bill_B]

                # TODO: do we save 2 bill to bill records for each comparison?
                bill_to_bill_new = pymodels.BillToBillModel(
                    billnumber_version= bill_A,
                    billnumber_version_to= bill_B,
                    score= from_doc_similarity[0][0],
                    bill_id= bill_id,
                    bill_to_id= bill_to_id
                )

                btb = utils_db.save_bill_to_bill(bill_to_bill_model= bill_to_bill_new, db=SessionLocal())

with Flow("Compare ES-related bills", executor=LocalDaskExecutor()) as flow:
    bills_to_compare = get_bills_to_compare()
    calculate_bill_similarity(bills_to_compare)

    
flow.register(project_name="BillSimilarityEngine")