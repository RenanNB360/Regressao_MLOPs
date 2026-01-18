import logging
from dotenv import load_dotenv
import dagshub

load_dotenv()

dagshub.init(
    repo_owner='RenanNB360',
    repo_name='Regressao_MLOPs',
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)
