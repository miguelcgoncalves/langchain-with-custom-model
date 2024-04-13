import requests


class PollinationsAIChat:
    url = "https://text.pollinations.ai/v1/chat/completions"
    model = "openai"

    def create_completions(self, **kwargs):
        payload = {
            **kwargs,
            "private": True,
            "model": self.model,
            "web_search": False,
            "seed": 1,
        }
        response = requests.post(
            self.url,
            json=payload,
        )
        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
        result = response.json()
        return result
