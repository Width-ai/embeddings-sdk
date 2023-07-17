import requests
from typing import List, Any

from domain import exceptions, models


class WidthSDKSession:
    def __init__(self, customer_id: str, api_key: str, url: str = "https://api.example.com"):
        self.api_url = url
        self.customer_id = customer_id
        self.api_key = api_key
        self.session = requests.Session()

        # Initialize session
        self.valid = False
        auth_resp = self.session.post(
            url=f"{self.api_url}/customer",
            data={
                'customer_id': self.customer_id,
                'api_key': self.api_key
            }
        )
        
        if auth_resp.status_code == 200 and auth_resp.json().get('exists', False):
            self.valid = True

    def keep_alive(self, model_version_id: str):
        """
        Keeps the container with model artifact alive and hot for quick inference
        """
        # TODO: need to implement logic on API side
        raise exceptions.NotImplemented

    def tear_down(self, model_version_id: str):
        """
        Stops the container with model artifact from staying alive
        """
        # TODO: need to implement logic on API side
        raise exceptions.NotImplemented

    def get_models(self) -> List[dict]:
        """
        Gets all models for the customer
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        models_resp = self.session.post(
            url=f"{self.api_url}/get_models",
            data={
                'customer_id': self.customer_id,
                'api_key': self.api_key
            }
        )
        
        if models_resp.status_code == 200:
            return models_resp.json()
        else:
            raise Exception(f"Error getting models: {models_resp.status_code}")

    def create_model(self, model_name: str) -> dict:
        """
        Creates a new model with the specified name and returns a dict with the id of the new model
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        model_resp = self.session.post(
            url=f"{self.api_url}/create_model",
            data={
                'customer_id': self.customer_id,
                'api_key': self.api_key,
                'model_name': model_name
            }
        )

        if model_resp.status_code == 200:
            return model_resp.json()
        else:
            raise Exception(f"Error creating model: {model_resp.status_code}")

    def finetune(
        self,
        model_id: str,
        datasets: List[models.FineTuneDataset],
        model_version_id: str = None,
        model_version: int = None,
        epochs: int = 10,
        batch_size: int = 4,
        evaluator: Any = None,
        learning_rate: float = 1e-3
    ):
        """
        Takes in model id and dataset to finetune new model version on

        Dataset should be in the following format:
        [
            {
                "loss": "tripletloss",
                "examples": [
                    ["Anchor 1", "Positive 1", "Negative 1"],
                    ["Anchor 2", "Positive 2", "Negative 2"]
                ]
            }
        ]
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        finetune_resp = self.session.post(
            url=f"{self.api_url}/finetune_model",
            data={
                'customer_id': self.customer_id,
                'api_key': self.api_key,
                'model_id': model_id,
                'model_version_id': model_version_id,
                'model_version': model_version,
                'datasets': datasets,
                'epochs': epochs,
                'batch_size': batch_size,
                'evaluator': evaluator,
                'learning_rate': learning_rate
            }
        )

        if finetune_resp.status_code == 200:
            return finetune_resp.json()
        else:
            raise Exception(f"Error finetuning: {finetune_resp.status_code}")

    def inference(self, model_id: str, model_version_id: str, input_texts: List[str]) -> List:
        """
        infers
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        inference_resp = self.session.post(
            url=f"{self.api_url}/inference",
            data={
                'customer_id': self.customer_id,
                'api_key': self.api_key,
                'model_id': model_id,
                'model_version_id': model_version_id,
                'input_texts': input_texts
            }
        )

        if inference_resp.status_code == 200:
            return inference_resp.json()
        else:
            raise Exception(f"Error performing inference: {inference_resp.status_code}")