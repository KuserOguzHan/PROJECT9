### 0. Information of Project

- The project aims to teach us how to use Jenkins and fastapi with the ML model.
- We will learn the CI-CD cycle in an ML PROJECT.
- First, we write code and send it to gitea, this process is CI.
- If the code is true, Jenkins will check the written code,  Jenkins will run new code.
- Thus, fastapi is to be run continuously.


### 1. Create Gitea Organization

### 2. Create Gitea Repository under Organization

### 3. Open the VisualStudio Code and Create project files

### 4. Create src/fastapi_hepsiburada_prediction

### 5. Create requirements.txt file
```
pandas<=1.4.1
scikit-learn<=1.0.2
joblib<=1.2.0
fastapi[all]==0.83.0
uvicorn[standard]==0.13.4
sqlmodel==0.0.8
pymysql==1.0.2
#python-dotenv==0.21.0
gunicorn==20.1.0
pytest==7.0.1
```

```
pip install -r requirements.txt
```

### 6. Create train.py file
- Thise file is ML a model.
- Creta a new file named "saved_models".
- Run this comond "python train.py"

```
import time
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def read_and_train():

    # read data
    df_origin = pd.read_csv("https://raw.githubusercontent.com/KuserOguzHan/mlops_1/main/hepsiburada.csv.csv")
    df_origin.head()

    df = df_origin.drop(["manufacturer"], axis=1)

    df.head()
    df.info()
    # Feature matrix
    X = df.iloc[:, 0:-1].values
    print(X.shape)
    print(X[:3])

    # Output variable
    y = df.iloc[:, -1]
    print(y.shape)
    print(y[:6])

    # split test train
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train model
    from sklearn.ensemble import RandomForestRegressor

    estimator = RandomForestRegressor(n_estimators=200)
    estimator.fit(X_train, y_train)

    # Test model
    y_pred = estimator.predict(X_test)
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print("R2: ".format(r2))

    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(r2)

    # Save Model
    import joblib
    joblib.dump(estimator, "saved_models/randomforest_with_hepsiburada.pkl")

    # make predictions
    # Read models
    estimator_loaded = joblib.load("saved_models/randomforest_with_hepsiburada.pkl")

    # Prediction set
    X_manual_test = [[64.0, 4.0, 6.50, 3500, 8.0, 48.0, 2.0, 2.0, 2.0]]
    print("X_manual_test", X_manual_test)

    prediction = estimator_loaded.predict(X_manual_test)
    print("prediction", prediction)

```

   
### 7. Create models.py
- Models.py is table of fastapi databases

```
from pydantic import BaseModel

class hepsiburada(BaseModel):
    memory: float
    ram: float
    screen_size: float
    power:float
    front_camera:float
    rc1:float
    rc3:float
    rc5:float
    rc7:float

    class Config:
        schema_extra = {
            "example": {
                "memory": 128.0,
                "ram": 8.0,
                "screen_size": 6.40,
                "power": 4310.0,
                "front_camera": 32.0,
                "rc1": 48.0,
                "rc3": 8.0 ,
                "rc5": 2.0,
                "rc7": 2.0,

            }
        }

```

### 8. Creta Main.py

- Main.py is the main file-code of fastapi.

```
import os
import pathlib
import joblib, uvicorn, argparse
from fastapi import FastAPI, Request

try:
    from models import hepsiburada
except:
    from fastapi_hepsiburada_prediction.models import hepsiburada


# Read models saved during train phase
current_dir = pathlib.Path(__file__).parent.resolve()
dirname = os.path.join(current_dir, 'saved_models')
estimator_hepsiburada_loaded = joblib.load(os.path.join(dirname, "randomforest_with_hepsiburada.pkl"))



app = FastAPI()

def make_hepsiburada_prediction(model, request):
    # parse input from request
    memory= request["memory"]
    ram= request["ram"]
    screen_size= request["screen_size"]
    power= request["power"]
    front_camera= request["front_camera"]
    rc1= request["rc1"]
    rc3= request["rc3"]
    rc5= request["rc5"]
    rc7= request["rc7"]


    # Make an input vector
    hepsiburada = [[memory, ram, screen_size, power, front_camera, rc1, rc3, rc5, rc7]]

    # Predict
    prediction = model.predict(hepsiburada)

    return prediction[0]

# Hepsiburada Prediction endpoint
@app.post("/prediction/hepsiburada")
def predict_hepsiburada(request: hepsiburada):
    prediction = make_hepsiburada_prediction(estimator_hepsiburada_loaded, request.dict())
    return {"result":prediction}

# Get client info
@app.get("/client")
def client_info(request: Request):
    client_host = request.client.host
    client_port = request.client.port
    return {"client_host": client_host,
            "client_port": client_port}

```
 
- Run uvicorn on src/fastapi_hepsiburada_prediction
```
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

### 9. Create test_main.py file
- It is related to main page of the test server interface

```
from fastapi.testclient import TestClient

try:
    from main import app
except:

    from fastapi_hepsiburada_prediction.main import app

client = TestClient(app)


def test_predict_hepsiburada():
    response = client.post("/prediction/hepsiburada", json={
        "memory": 128.0,
        "ram": 8.0,
        "screen_size": 6.40,
        "power": 4310.0,
        "front_camera": 32.0,
        "rc1": 48.0,
        "rc3": 8.0,
        "rc5": 2.0,
        "rc7": 2.0
    })

    assert response.status_code == 200
    assert isinstance(response.json()['result'], float), 'Result wrong type!'

```


### 10. Create run_train.py file

- Aim to run easy.

```
import train

if __name__ == '__main__':
    train.read_and_train()
    
```

### 11. Create playbooks folder and move src file to inside it

``` 
└── playbooks
    ├── prod
    ├── src/fastapi_hepsiburada_prediction
    ├── test

``` 


### 12. Create Jenkinsfile

- I want to learn whether Jenkis is runing.

```
├── playbooks
├── Jenkinsfile
├── README.md
├── hosts
```

``` 
pipeline{
   agent any
   stages{
        stage(" Test") {
           steps {
                sh 'echo "Hellooooooooooooooooooooooooooooooooooo"'
            }
        }

     }
}
``` 


### 13. Create install-fast-on-test.yaml file

``` 
└── playbooks
    ├── prod
    ├── src/fastapi_hepsiburada_prediction
    ├── test
    ├── install-fast-on-test.yaml

``` 
- install-fast-on-test.yaml file is to create a test server.

``` 
- hosts: test
  become: yes
  tasks:
    - name: Install rsync
      yum:
        name: rsync
        state: latest

    - name: Copy files to remote server
      synchronize:
        src: src
        dest: /opt/fastapi
``` 

### 14. Change Jenkinsfile

```
├── playbooks
├── Jenkinsfile
├── README.md
├── hosts
```

``` 
pipeline{
   agent any
   stages{
        stage(" Install FastAPI on Test Server") {
           steps {
                ansiblePlaybook credentialsId: 'jenkins_pk', disableHostKeyChecking: true, installation: 'Ansible',
                inventory: 'hosts', playbook: 'playbooks/install-fast-on-test.yaml'
            } 
        }
    }

} 
``` 

### 15. Create hosts file

- test ve prod serverlerın ip adresleri ve connection türü, user türü gibi bilgiler

```
├── playbooks
├── Jenkinsfile
├── README.md
├── hosts
```

```
[all:vars]
ansible_ssh_common_args='-o StrictHostKeyChecking=no'
ansible_connection = ssh
ansible_python_interpreter = /usr/bin/python3

[172.18.0.8]
test ansible_host=test ansible_user=test_user

[172.18.0.7]
prod ansible_host=prod ansible_user=prod_user
```

### 16. Check if the files copied to the test_server

- Oluşturduğumuz Jenkinsfile dosyasının içeriğinde olan bilgilerin kopyalanıp kopyalanmadığını kontrol ediyoruz.
``` 
(fastapi) [train@localhost fastapi_hepsiburada_prediction]$ docker exec -it test_server bash
``` 
```
[root@test_server /]# ls -l /opt/
[root@test_server /]# ls -l /opt/fastapi/src/fastapi_advertising_prediction/
```

### 17. Update install-fast-on-test.yaml file
- Servicefileın kopyalanması,pip kurulumu, requirements indirilmesi eklendi

``` 
- hosts: test
  become: yes
  tasks:
    - name: Install rsync
      yum:
        name: rsync
        state: latest

    - name: Copy files to remote server
      synchronize:
        src: src
        dest: /opt/fastapi

    - name: Copy service file
      synchronize:
        src: test/fastapi.service
        dest: /etc/systemd/system/fastapi.service

    - name: Upgrade pip
      pip:
        name: pip
        state: latest
        executable: pip3

    - name: Install pip requirements
      pip:
        requirements: /opt/fastapi/src/fastapi_hepsiburada_prediction/requirements.txt
``` 

### 18. Update install-fast-on-test.yaml file with the last changes

``` 
- hosts: test
  become: yes
  tasks:
    - name: Install rsync
      yum:
        name: rsync
        state: latest

    - name: Copy files to remote server
      synchronize:
        src: src
        dest: /opt/fastapi

    - name: Copy service file
      synchronize:
        src: test/fastapi.service
        dest: /etc/systemd/system/fastapi.service

    - name: Upgrade pip
      pip:
        name: pip
        state: latest
        executable: pip3

    - name: Install pip requirements
      pip:
        requirements: /opt/fastapi/src/fastapi_hepsiburada_prediction/requirements.txt

    - name: Env variables for fastapi
      shell: |
        export LC_ALL=en_US.utf-8
        export LANG=en_US.utf-8

    - name: Check if Service Exists
      stat: path=/etc/systemd/system/fastapi.service
      register: service_status

    - name: Stop Service
      service: name=fastapi state=stopped
      when: service_status.stat.exists
      register: service_stopped

    - name: Start fastapi
      systemd:
        name: fastapi
        daemon_reload: yes
        state: started
        enabled: yes
``` 
#check
```
[root@test_server /]# systemctl status fastapi
```
#check test server 

```
localhost:8001/docs
```
### 19. Update Jenkinsfile
``` 
pipeline{
   agent any
   stages{
        stage(" Install FastAPI on Test Server") {
           steps {
                ansiblePlaybook credentialsId: 'jenkins_pk', disableHostKeyChecking: true, installation: 'Ansible',
                inventory: 'hosts', playbook: 'playbooks/install-fast-on-test.yaml'
            } 
        }
        stage(" Test FastAPI on Test Server") {
           steps {
                ansiblePlaybook credentialsId: 'jenkins_pk', disableHostKeyChecking: true, installation: 'Ansible',
                inventory: 'hosts', playbook: 'playbooks/testing-fastapi.yaml'
            }
        }        
        
        
    }

} 
``` 
### 20. Create testing-fastapi.yaml file

``` 
└── playbooks
    ├── src/fastapi_hepsiburada_prediction
    ├── testing-fastapi.yaml

``` 

``` 
- hosts: test
  become: yes
  tasks:
    - name: Install rsync
      yum:
        name: rsync
        state: latest

    - name: Copy files to remote server
      synchronize:
        src: src
        dest: /opt/fastapi

    - name: Upgrade pip
      pip:
        name: pip
        state: latest
        executable: pip3

    - name: Install pip requirements
      pip:
        requirements: /opt/fastapi/src/fastapi_hepsiburada_prediction/requirements.txt

    - name: Env variables for fastapi
      shell: |
        export LC_ALL=en_US.utf-8
        export LANG=en_US.utf-8
    - name: Run Test script
      command: bash -c 'cd /opt/fastapi/src/fastapi_hepsiburada_prediction/ && /usr/local/bin/pytest'

``` 

### 21. Update Jenkinsfile with the last changes

### 22. Create install-fast-on-prod.yaml file

``` 
└── playbooks
    ├── src/fastapi_hepsiburada_prediction
    ├── testing-fastapi.yaml
    ├── install-fast-on-prod.yaml

``` 

```
- hosts: prod
  become: yes
  tasks:
    - name: Install rsync
      yum:
        name: rsync
        state: latest

    - name: Copy files to remote server
      synchronize:
        src: src
        dest: /opt/fastapi

    - name: Copy service file
      synchronize:
        src: prod/fastapi.service
        dest: /etc/systemd/system/fastapi.service

    - name: Upgrade pip
      pip:
        name: pip
        state: latest
        executable: pip3

    - name: Install pip requirements
      pip:
        requirements: /opt/fastapi/src/fastapi_hepsiburada_prediction/requirements.txt

    - name: Env variables for fastapi
      shell: |
        export LC_ALL=en_US.utf-8
        export LANG=en_US.utf-8
    - name: Check if Service Exists
      stat: path=/etc/systemd/system/fastapi.service
      register: service_status

    - name: Stop Service
      service: name=fastapi state=stopped
      when: service_status.stat.exists
      register: service_stopped

    - name: Start fastapi
      systemd:
        name: fastapi
        daemon_reload: yes
        state: started
        enabled: yes
```



### 23.1. Create prod folder

``` 
└── playbooks
    ├── prod
        └── fastapi.service
    ├    
    ├── src/fastapi_hepsiburada_prediction
    ├── test
    ├── install-fast-on-test.yaml
    
```
### 23.2. Create fastapi.service file under the prod directory

```
[Unit]
Description=Gunicorn
Documentation=https://docs.gunicorn.org/en/stable/deploy.html

[Service]
Type=simple
ExecStart=/bin/bash -c 'cd /opt/fastapi/src/fastapi_hepsiburada_prediction/ && /usr/local/bin/gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000'
ExecStop=pkill -f python3

[Install]
WantedBy=multi-user.target
```

### 24.Chechc

- check test: localhost:8001/docs
- check prod: localhost:8000/docs
- 
