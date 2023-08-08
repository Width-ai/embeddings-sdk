# WordEmbeddings SDK
Python sdk to interface with the Word Embeddings API


## Example Usage
```python
from word_embeddings_sdk import WordEmbeddingsSession
session = WordEmbeddingsSession(customer_id="...", api_key="...", url="https://wordembeddings.ai")
session.get_models()
model_info = session.create_model(model_name="testing model")
session.get_models()
```
