import langchain
import os
print(f"LangChain file: {langchain.__file__}")
print(f"LangChain dir: {os.path.dirname(langchain.__file__)}")
try:
    import langchain.chains
    print("langchain.chains found")
except ImportError as e:
    print(f"langchain.chains NOT found: {e}")

try:
    from langchain.chains import RetrievalQA
    print("RetrievalQA found")
except ImportError as e:
    print(f"RetrievalQA NOT found: {e}")
