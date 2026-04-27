from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="AIzaSyBfgTIsw3n7RYSNdNwjhIp4jEBvdpj1ws8")

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="o borreguitos é debochado mesmo?"
)
print(response.text)