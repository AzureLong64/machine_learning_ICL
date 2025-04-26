# pipeline/client/client.py
from openai import OpenAI

class APIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url

    def get_client(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
