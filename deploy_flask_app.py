import os
from pythonanywhere.api import Webapp
from pythonanywhere import AuthenticatedUser

def deploy():
    username = os.environ['PA_USERNAME']
    api_token = os.environ['PA_API_TOKEN']
    domain_name = 'zackam.pythonanywhere.com'

    user = AuthenticatedUser(username, api_token)
    webapp = user.webapps.get(domain_name)
    webapp.reload()

if __name__ == "__main__":
    deploy()