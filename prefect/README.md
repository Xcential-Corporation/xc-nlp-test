# Startup instructions

```
docker-compose up
prefect backend server
prefect create project BillSimilarityEngine 
prefect server create-tenant --name default --slug default
python3 training.py
prefect agent local start
```

The UI will be available on localhost:8080.

Every time you make changes to a flow, you have to run `python3 training.py` to register the new changes to Prefect.