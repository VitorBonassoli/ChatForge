import os
from dotenv import load_dotenv
from google import genai  # Importação correta para esta biblioteca

load_dotenv()

# Em vez de .configure(), você cria um cliente
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Para usar, você chama o cliente
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Olá, tudo bem?"
)

print(response.text)