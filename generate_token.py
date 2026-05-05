import os.path
import glob
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Find the client secret file
            client_secret_file = 'credentials.json'
            if not os.path.exists(client_secret_file):
                json_files = glob.glob('client_secret_*.json')
                if json_files:
                    client_secret_file = json_files[0]
                else:
                    print("Error: Could not find credentials.json or client_secret_*.json")
                    return

            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_file, SCOPES)
            # This will open a browser window for you to log in
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
        print("\n✅ Success! token.json has been created.")
        print("You can now safely delete the client_secret JSON file if you want, but keep token.json safe!")

if __name__ == '__main__':
    main()
