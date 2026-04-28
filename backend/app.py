from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="AIzaSyBfgTIsw3n7RYSNdNwjhIp4jEBvdpj1ws8")

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Você sera um chat de vendas automatico inteligente, responda o clinente com clareza sem estender muito as respostas, informações, responda com o que o cliente esta buscando, caso o cliente relate um problema, retorene um produto que pode ajudar com o problem dele"
)
print(response.text)