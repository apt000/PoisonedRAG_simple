import os
from openai import OpenAI

XAI_API_KEY = "xai-2VrLMXu54ZJvcsVOENfaQPLDOWkTMx9NVVFGhawdvXiDYsGt0qlqooQmvcfD4jiYevaRwubDEy0aQKjA"
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

'''completion = client.chat.completions.create(
    model="grok-beta",
    messages=[
        {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."},
        {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
    ],
)'''

# print(completion.choices[0].message.content)

def get_res(prompt):
    completion = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

