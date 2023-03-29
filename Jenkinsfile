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

        stage(" Install FastAPI on Prod Server") {
           steps {
                ansiblePlaybook credentialsId: 'jenkins_pk', disableHostKeyChecking: true, installation: 'Ansible',
                inventory: 'hosts', playbook: 'playbooks/install-fast-on-prod.yaml'
            }
        }
     }
}