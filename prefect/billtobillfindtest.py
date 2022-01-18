import re

import billsim

from sqlalchemy.orm import Session
from billsim.utils import getDefaultNamespace, getBillLength, getBillLengthbyPath, getBillnumberversionParts, getId, getEnum, getText
from billsim.database import SessionLocal

from billsim import utils_db, pymodels


# Collect the bills to run comparisons on

with SessionLocal() as session:
    rs = session.execute("SELECT billtobill.bill_id, billtobill.bill_to_id, billtobill.score_es, bill.billnumber, bill.version FROM billtobill, bill WHERE billtobill.score_es IS NOT NULL AND (billtobill.bill_id = bill.id) ORDER BY billtobill.bill_id ASC, billtobill.score_es DESC;")
    results = rs.fetchall()

    print("Number of records with value for ES score: ", len(results))
    # group by bill_id
    bill_id_obj = {}
    for result in results:
        bill_id = result[0]
        bill_to_id = result[1]
        score_es = result[2]
        billnumber = result[3]
        billversion = result[4]

        if not bill_id in bill_id_obj.keys():
            bill_id_obj[bill_id] = []

        # Format results like so: { 10315 : [[10315, 631.2436789999999, '116hr5991', 'ih'] }
        bill_id_obj[bill_id].append([bill_to_id, score_es, billnumber, billversion])

    # keep only the top 30 values for es_score for each bill_id  
    new_bill_id_obj = {}

    for key, value in bill_id_obj.items():
        if len(value) < 30 and len(value) > 10:
            new_bill_id_obj[key] = value
        else:
            new_bill_id_obj[key] = value[0:30]
        
    count = 0
    for val in new_bill_id_obj.values():
        count = count + len(val)


    print("Number of bills to compare: ", len(new_bill_id_obj.keys()))
    print("Total number of comparisons to run: ", count)

