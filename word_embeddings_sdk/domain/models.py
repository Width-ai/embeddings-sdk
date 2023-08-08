from pydantic import BaseModel
from typing import List

class FineTuneDataset(BaseModel):
    loss: str = "tripletloss"
    examples: List