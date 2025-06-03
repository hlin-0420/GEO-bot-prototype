import os
import re
import sys
import pytest

# Configure environment before importing the package
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Data', 'html_files', 'visualization', 'Lines')
)
os.environ['HTM_FOLDER_PATH'] = DATA_DIR

# Make src.config importable as 'config'
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.config as src_config
sys.modules['config'] = src_config

import src.load_htm_file as load_htm_file
from src.rag_pipeline import build_offline_chatbot

# Patch BeautifulSoup so load_htm_file works without bs4
_orig_bs = load_htm_file.BeautifulSoup

def _patched_bs(obj, parser="html.parser"):
    if hasattr(obj, 'read'):
        obj = obj.read()
    return _orig_bs(obj, parser)

load_htm_file.BeautifulSoup = _patched_bs

@pytest.fixture(scope="module")
def qa_chain():
    content = load_htm_file.load_all_htm_files_content()
    content = re.sub(r"\n\s*\n", "\n", content)
    return build_offline_chatbot(content)

@pytest.mark.parametrize(
    "question",
    [
        "How do I add a line?",
        "How do I delete a line?",
    ],
)
def test_queries_return_answers(qa_chain, question):
    response = qa_chain.invoke({"input": question})
    assert isinstance(response, dict)
    assert response.get("answer")
    assert response["answer"].strip() != ""
