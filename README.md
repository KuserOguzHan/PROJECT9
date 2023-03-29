### 1. Create Gitea Organization
### 2. Create Gitea Repository under Organization
### 3. Open the VisualStudio Code and Create project files
### 4. Create src/fastapi_hepsiburada_prediction
### 5. Create train.py file
### 6. Create requirements.txt file
### 7. pip install -r requirements.txt
### 8. mkdir saved_models
### 9. run train.py : python train.py
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


read_and_train()

```    
### 10. Create models.py, run_train.py, main.py, Dockerfile

### 11. Check fastapi with uvicorn
``` 
  uvicorn main:app --host 0.0.0.0 --port 8002 --reload
``` 

### 12. Create test_main.py file

### 13. run uvicorn on src/fastapi_hepsiburada_prediction ..

### 14. run test_main.py

 ``` 
 pytest
 ``` 

### 15. create new Jenkins item.

### 16. Create Jenkinsfile
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
### 17. Create playbooks folder and move src file to inside it

### 18. Create install-fast-on-test.yaml file

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

### 19. Change Jenkinsfile
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

### 20. Create hosts file

### 21. Check if the files copied to the test_server

``` 
(fastapi) [train@localhost fastapi_hepsiburada_prediction]$ docker exec -it test_server bash
``` 
```
[root@test_server /]# ls -l /opt/
[root@test_server /]# ls -l /opt/fastapi/src/fastapi_advertising_prediction/
```

### 22. Update install-fast-on-test.yaml file

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

### 23. Update install-fast-on-test.yaml file with the last changes

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
### 24. Update Jenkinsfile
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
### 25. Create testing-fastapi.yaml file

### 26. Update Jenkinsfile with the last changes

### 27. Create install-fast-on.yaml file

### 28. Create prod folder

### 29. Create fastapi.service file under the prod directory

- check test: localhost:8001/docs
- check prod: localhost:8000/docs
- 
