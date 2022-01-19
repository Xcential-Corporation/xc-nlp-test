# Startup instructions

### Install requirements
```
pip install -r requirements.txt
```

### Bring up the docker containers
```
docker-compose up
```

### Set up Prefect
```
prefect backend server
prefect server create-tenant --name default --slug default
prefect create project BillSimilarityEngine 
```


### Register

Prefect requires you to register your flows for versioning and scheduling purposes. Our flow files contain a call to `prefect.register` as their last line of code, so you just have to run the file to register the flow with Prefect each time you make a change:

```
python FILENAME.py
```

Every time you make changes to a flow, you have to run `python FILENAME.py` to register the new version of your flow. 

To enable parallelization, Prefect flows are run by agents that are decoupled from its UI/backend. To start an agent local to your machine:
```
prefect agent local start
```

The UI will be available on localhost:8080. You can navigate there, find a flow flow, and run it.

# Inspect data

Access the billsim Postgres DB like so:

```
psql -h localhost -p 5433 -d postgres -U postgres
```