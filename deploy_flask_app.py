import os
from pythonanywhere.api import PythonAnywhere, Webapp

def deploy():
    username = os.environ['PA_USERNAME']
    api_token = os.environ['PA_API_TOKEN']
    domain_name = 'zackam.pythonanywhere.com'

    pa = PythonAnywhere(username, api_token)
    webapp = pa.webapps.get(domain_name)
    webapp.reload()

if __name__ == "__main__":
    deploy()