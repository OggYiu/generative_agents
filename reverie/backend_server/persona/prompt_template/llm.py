# Import the necessary module
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import List

# Define the LLM class
class LLM:
    def __init__(self):
        pass

    def generate_response(
            self,
            model: str,
            prompt: str,
            system_prompt: str = '',
            max_tokens: int = None,
            temperature: float = 0.0,
            top_p: float = None,
            stream: bool = None,
            frequency_penalty: float = None,
            presence_penalty: float = None,
            stop: List[str] = None):
        try:
            llm = ChatOpenAI(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                # stream=stream, # not avaliable in langchain
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop)
            messages = [
                ("system", system_prompt),
                ("human", prompt),
            ]
            ai_msg = llm.invoke(messages)
            return ai_msg.content
        except Exception as e: 
            raise Exception(e)
            # print (e)
            # return e
            
    def embeddings(self, model: str,text: str):
        try:
            embeddings = OpenAIEmbeddings(model=model)
            return embeddings.embed_query(text)
        except Exception as e:
            raise Exception(e)
            # print (e)
            # return e