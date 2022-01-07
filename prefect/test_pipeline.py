import pickle
import time
from os import path, listdir
from pathlib import Path
from os.path import isfile, join
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import csv
import sqlite3
import re

import billsim
from billsim import utils_db, pymodels


con = sqlite3.connect('example.db')
cur = con.cursor()

#transform document into vectorized space
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

# def calculate_doc_similarity(A_doc, B_doc, A_section_doc, B_section_doc, doc_tfidf_vectorizer, sec_tfidf_vectorizer):
#     A_section_doc_vectorized = section_doc_tfidf_vectorized_transformation(A_section_doc, sec_tfidf_vectorizer)
#     B_section_doc_vectorized = section_doc_tfidf_vectorized_transformation(B_section_doc, sec_tfidf_vectorizer)


#     # print('Scores and timing')
#     # print(elapsed_1, doc_sim_score)

#     elapsed_2, sec_doc_sim_score = cosine_pairwise_sim(A_section_doc_vectorized, B_section_doc_vectorized)

#     # print(elapsed_2, sec_doc_sim_score)
#     print("Fin")

#     return doc_sim_score, sec_doc_sim_score


def calculate_doc_similarity(A_doc, B_doc, doc_tfidf_vectorizer):
    A_doc_vectorized = document_tfidf_vectorized_transformation(A_doc, doc_tfidf_vectorizer)
    B_doc_vectorized = document_tfidf_vectorized_transformation(B_doc, doc_tfidf_vectorizer)

    elapsed_1, doc_sim_score = cosine_pairwise_sim(A_doc_vectorized, B_doc_vectorized)

    return doc_sim_score

start = time.time()
print('Opening files...')

section_corpus_data = open("tv_section_corpus_data.pickel", "rb")
doc_corpus_data = open("tv_doc_corpus_data.pickel", "rb")
doc_tfidf_vectorizer = open("document_tfidf_vectorizer.pickel", "rb")
sec_tfidf_vectorizer = open("section_tfidf_vectorizer.pickel", "rb")

section_corpus_data = pickle.load(section_corpus_data)
doc_corpus_data = pickle.load(doc_corpus_data)
doc_tfidf_vectorizer = pickle.load(doc_tfidf_vectorizer)
sec_tfidf_vectorizer = pickle.load(sec_tfidf_vectorizer)

# print(section_corpus_data[570])
# print(doc_corpus_data[570])

done = time.time()
elapsed = done - start
print('Opening files took:', elapsed)

A_doc_name = 'BILLS-117hr1319enr'
B_doc_name = 'BILLS-117hr1319eh' 

A_doc = [i[1] for i in doc_corpus_data if A_doc_name ==i[0]][0]
B_doc = [i[1] for i in doc_corpus_data if B_doc_name ==i[0]][0]

A_section_doc = [i[2] for i in section_corpus_data if A_doc_name ==i[0]]
B_section_doc = [i[2] for i in section_corpus_data if B_doc_name ==i[0]]

print('Calculating similarity...')

start = time.time()

doc_sim_results = []

for item_a in doc_corpus_data[700:750]:
    for item_b in doc_corpus_data[700:750]:

        compare_start = time.time()
        A_doc = item_a[1]
        B_doc = item_b[1]
        from_doc_similarity = calculate_doc_similarity(A_doc, B_doc, doc_tfidf_vectorizer)
        to_doc_similarity = calculate_doc_similarity(B_doc, A_doc, doc_tfidf_vectorizer)
        compare_end = time.time()
        elapsed = compare_end - compare_start

        bill_a = re.sub("BILLS-", "", item_a[0])
        bill_b = re.sub("BILLS-", "", item_b[0])
        # save to db
        ids = utils_db.get_bill_ids([bill_a, bill_b])

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
            # print("Skipping save.")
            # print(btb)


        # print("Compared ", item_a[0], "with", item_b[0], "and found result: ", from_doc_similarity[0][0])
        result = {"bill1": item_a[0], "bill2": item_b[0], "doc_level_similarity": from_doc_similarity[0][0]}


        doc_sim_results.append(result)

print('Doc corpus length', len(doc_corpus_data))

def sort_by_similarity(result):
    return result["doc_level_similarity"]

doc_sim_results.sort(key=sort_by_similarity, reverse=True)


print("Writing results to file")

data_file = open(f'Bill Similarity Results - {time.asctime()}', 'w', newline='')
csv_writer = csv.writer(data_file)

count = 0
for result in doc_sim_results:
    if count == 0:
        header = result.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(result.values())

data_file.close()

# # print('Doc sim results length', len(doc_sim_results))
# print('Sample results')
# print(doc_sim_results)
# print(doc_sim_results[157])
# print('Underlying data')
# print(doc_corpus_data[157])
print('Done')

done = time.time()
elapsed = done - start
print('Calculating similarity took:', elapsed)



