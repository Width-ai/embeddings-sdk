# Embeddings SDK
Python sdk to interface with Width.Ai's Embeddings SaaS


## Example Usage
```python
from embeddings_sdk import WidthEmbeddingsSession
session = WidthEmbeddingsSession(customer_id="...", api_key="...", url="https://embeddings.width.ai")
session.get_models()
model_info = session.create_model(model_name="testing model")
session.get_models()
```
