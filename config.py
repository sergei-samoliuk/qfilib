import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CALYPSO_LOGIN = os.getenv('CALYPSO_LOGIN')
CALYPSO_PASSWORD = os.getenv('CALYPSO_PASSWORD')

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
DATA_INPUT_DIR = DATA_DIR / 'input'
LUIGI_OUTPUT_DIR = DATA_DIR / 'output'

def setup_proxies():
    os.environ['http_proxy'] = "http://msk-nrbc-proxy:3128"
    os.environ['https_proxy'] = "http://msk-nrbc-proxy:3128"


