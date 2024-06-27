import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from typing import List, Tuple
from config import config
import json

logger = logging.getLogger('master')
chat_logger = logging.getLogger('chat')

class Agent:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.chat_model = ChatOpenAI(model_name=config.MODEL_NAME)

    async def process_query(self, query: str) -> str:
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(query)
        context = "\n\n".join([f"{doc[0]}\n{doc[1]} ({doc[2]})" for doc in relevant_memories])
        
        messages = [
            HumanMessage(content=f"REQUEST = \"{query}\". CONTEXT = \"\"\"{context}\"\"\" INSTRUCTIONS = Your 'CONTEXT' is formed of reverse chronologically ordered 'memories' in the format <REQUEST>,<RESPONSE> (<TIMESTAMP>). Use these to best respond to the request.")
        ]
        
        # Log the full message sent to the API
        chat_logger.info(f"Full ChatGPT API request: {json.dumps([m.dict() for m in messages], indent=2)}")
        
        response = await self.chat_model.ainvoke(messages)
        
        # Log the full API response
        chat_logger.info(f"Full ChatGPT API response: {json.dumps(response.dict(), indent=2)}")
        
        logger.debug(f"Processed query: {query} with context: {context}")
        
        return response.content