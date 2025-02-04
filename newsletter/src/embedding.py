from openai import OpenAI


class TextEmbedder:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding