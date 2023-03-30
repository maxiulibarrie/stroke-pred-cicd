import os
from base64 import b64decode

def decode_save_credentials():
    key = os.environ.get('SERVICE_ACCOUNT_KEY')
    gs_credential = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    with open(gs_credential,'w') as json_file:
        json_file.write(b64decode(key).decode())
    print(os.path.realpath(gs_credential))

if __name__ == '__main__':
    decode_save_credentials()
    