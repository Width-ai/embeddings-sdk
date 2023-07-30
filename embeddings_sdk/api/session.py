import json
import requests
from typing import List, Any

from embeddings_sdk.domain import exceptions, models
from embeddings_sdk.utils.utils import setup_logger


class WidthEmbeddingsSession:
    def __init__(self, customer_id: str, api_key: str, url: str = "https://api.example.com"):
        self.logger = setup_logger(__name__)
        self.api_url = url
        self.customer_id = customer_id
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'customer_id': self.customer_id,
            'api_key': self.api_key
        })

        # Initialize session
        self.valid = False
        auth_resp = self.session.get(url=f"{self.api_url}/customer")
        
        if auth_resp.status_code == 200 and auth_resp.json().get('exists', False):
            self.valid = True
        else:
            self.logger.warning(f"Session not authenticated, check api_key and customer_id are valid")

    def keep_alive(self, model_id: str, model_version_id: str) -> bool:
        """
        Keeps the container with model artifact alive and hot for quick inference
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")

        keep_alive_resp = self.session.post(
            url=f"{self.api_url}/keep_alive",
            data=json.dumps({
                'model_id': model_id,
                'model_version_id': model_version_id
            })
        )

        if keep_alive_resp.ok:
            self.logger.info(
                "Model worker will be kept alive until tear down specified. "
                f"worker info:\nmodel_id: {model_id}\nmodel_version_id: {model_version_id}"
            )
            return True
        else:
            self.logger.warning(
                "Issue setting worker to be kept alive: status code: "
                f"{keep_alive_resp.status_code}, text: {keep_alive_resp.text}"
            )
        return False


    def tear_down(self, model_id: str, model_version_id: str) -> bool:
        """
        Stops the container with model artifact from staying alive
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")

        tear_down_resp = self.session.post(
            url=f"{self.api_url}/tear_down",
            data=json.dumps({
                'model_id': model_id,
                'model_version_id': model_version_id
            })
        )

        if tear_down_resp.ok:
            self.logger.info(
                "Model worker will be torn down. "
                f"worker info:\nmodel_id: {model_id}\nmodel_version_id: {model_version_id}"
            )
        else:
            self.logger.warning(f"Issue setting worker to be kept alive: status code: {tear_down_resp.status_code}, text: {tear_down_resp.text}")

    def get_models(self) -> List[dict]:
        """
        Gets all models for the customer
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        models_resp = self.session.get(url=f"{self.api_url}/models")
        
        if models_resp.ok:
            return models_resp.json()
        else:
            raise Exception(f"Error getting models: {models_resp.status_code} {models_resp.text}")

    def create_model(self, model_name: str) -> dict:
        """
        Creates a new model with the specified name and returns a dict with the id of the new model
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        model_resp = self.session.post(
            url=f"{self.api_url}/model",
            data=json.dumps({
                'model_name': model_name
            })
        )

        if model_resp.ok:
            return model_resp.json()
        else:
            raise Exception(f"Error creating model: {model_resp.status_code} {model_resp.text}")

    def delete_model(self, model_id: str) -> bool:
        """
        Deletes a model by id
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        model_resp = self.session.delete(
            url=f"{self.api_url}/model/{model_id}",
        )

        if model_resp.ok:
            return True
        else:
            raise Exception(f"Error deleting model: {model_resp.status_code} {model_resp.text}")

    def delete_model_version(self, model_id: str, model_version_id: str) -> bool:
        """
        Deletes a model version by id
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        model_resp = self.session.delete(
            url=f"{self.api_url}/model/{model_id}/model_version/{model_version_id}",
        )

        if model_resp.ok:
            return True
        else:
            raise Exception(f"Error deleting model version: {model_resp.status_code} {model_resp.text}")

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
    ) -> dict:
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
            data=json.dumps({
                'model_id': model_id,
                'model_version_id': model_version_id,
                'model_version': model_version,
                'datasets': datasets,
                'epochs': epochs,
                'batch_size': batch_size,
                'evaluator': evaluator,
                'learning_rate': learning_rate
            })
        )

        if finetune_resp.ok:
            return finetune_resp.json()
        else:
            raise Exception(f"Error finetuning: {finetune_resp.status_code} {finetune_resp.text}")

    def check_status(self, finetune_id: str) -> dict:
        """
        get the status of an finetuning job
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        status_resp = self.session.post(
            url=f"{self.api_url}/status_model",
            data=json.dumps({
                'finetune_id': finetune_id
            })
        )

        if status_resp.ok:
            return status_resp.json()
        else:
            raise Exception(f"Error performing inference: {status_resp.status_code} {status_resp.text}")

    def get_model_versions(self, model_id: str = None, model_version_id: str = None) -> List[dict]:
        """
        Get all model versions belonging to a customer
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        data = {
            'model_id': model_id,
            'model_version_id': model_version_id
        }
        # filter out any nones
        data = {k: v for k, v in data.items() if v}

        model_versions_resp = self.session.post(
            url=f"{self.api_url}/model_versions",
            data=json.dumps(data)
        )

        if model_versions_resp.ok:
            return model_versions_resp.json()
        else:
            raise Exception(f"Error getting model versions: {model_versions_resp.status_code} {model_versions_resp.text}")

    def inference(self, model_id: str, model_version_id: str, input_texts: List[str]) -> List:
        """
        infers
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        inference_resp = self.session.post(
            url=f"{self.api_url}/inference",
            data=json.dumps({
                'model_id': model_id,
                'model_version_id': model_version_id,
                'input_texts': input_texts
            })
        )

        if inference_resp.ok:
            return inference_resp.json()
        else:
            raise Exception(f"Error performing inference: {inference_resp.status_code} {inference_resp.text}")

    def evaluate(self, model_id: str, model_version_id: str, samples: List, similarity_function: str = "cosine") -> List:
        """
        evaluates
        """
        if not self.valid:
            raise exceptions.InvalidAPICredentials("Invalid API session. Please check your credentials.")
        
        evaluation_resp = self.session.post(
            url=f"{self.api_url}/evaluation",
            data=json.dumps({
                'model_id': model_id,
                'model_version_id': model_version_id,
                'samples': samples,
                'similarity_function': similarity_function,
            })
        )

        if evaluation_resp.ok:
            return evaluation_resp.json()
        else:
            raise Exception(f"Error performing evaluation: {evaluation_resp.status_code} {evaluation_resp.text}")