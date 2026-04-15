from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def queryOpenAI(prompt, sysRole = None, model = "gpt-4o-mini", temperature = 0):
    if sysRole == None:
        sysRole = "Follow the instructions given in user carefully"
    response = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "system", "content": sysRole},
            {"role": "user", "content": prompt}
        ],
        temperature = temperature
    )
    return response.choices[0].message.content.strip()