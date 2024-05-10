from pythonanywhere.api import Webapp

def deploy():
    username = os.environ['PA_USERNAME']
    api_token = os.environ['PA_API_TOKEN']
    domain_name = 'zackam.pythonanywhere.com'

    webapp = Webapp(username, domain_name, api_token)
    webapp.reload()

if __name__ == "__main__":
    deploy()
