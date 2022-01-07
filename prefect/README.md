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

Prefect requires you to register your flows for versioning and scheduling purposes. `training.py` contains a call to `prefect.register`. To register the training flow with Prefect:

```
python training.py
```

Every time you make changes to a flow, you have to run `python3 training.py` to register the new version of your flow. 

To enable parallelization, Prefect flows are run by agents that are decoupled from its UI/blackend. To start an agent local to your machine:
```
prefect agent local start
```

The UI will be available on localhost:8080. You can navigate there, find the "Training" flow, and run it.

