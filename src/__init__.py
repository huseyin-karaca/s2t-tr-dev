import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# test the dummy env variable
test_var = os.getenv("TEST_ENV_VAR")

# Duruma göre log yazdır
if test_var != "Hello_from_dotenv":
    logger.error(f"Failed to load environment variables! TEST_ENV_VAR returned: {test_var}")
