from utils import *
import ollama

# import openai
# openai.api_key = openai_api_key

from openai import OpenAI
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

def llm_chat_create(model, prompt):
#   completion = openai.ChatCompletion.create(
#     model=llm_model_cheap, 
#     messages=[{"role": "user", "content": prompt}]
#   )
#   return completion["choices"][0]["message"]["content"]

  completion = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "user", "content": prompt}
    ]
  )
  return completion.choices[0].message.content


def llm_completion_create(model, prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stream, stop):
    
    # response = llm_completion_create(
    #             model=gpt_parameter["engine"],
    #             prompt=prompt,
    #             temperature=gpt_parameter["temperature"],
    #             max_tokens=gpt_parameter["max_tokens"],
    #             top_p=gpt_parameter["top_p"],
    #             frequency_penalty=gpt_parameter["frequency_penalty"],
    #             presence_penalty=gpt_parameter["presence_penalty"],
    #             stream=gpt_parameter["stream"],
    #             stop=gpt_parameter["stop"],)
    # return response.choices[0].text

    completion = client.chat.completions.create(
        model=model,
        messages=[
        {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stream=stream,
        stop=stop,
    )
    return completion.choices[0].message.content

def llm_embedding_create(model, text):
    # openai.Embedding.create(
    #         input=[text], model=model)['data'][0]['embedding']

  return ollama.embeddings(
    prompt=text, model=model)['embedding']