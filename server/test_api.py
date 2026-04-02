from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

try:
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=50
    )

    print("  API WORKING!")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("  API ERROR")
    print(e)