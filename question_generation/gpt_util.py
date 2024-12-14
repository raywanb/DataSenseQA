import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict
#os.environ["OPENAI_API_KEY"] =
load_dotenv()

# Retrieve the API key from the .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#conversation_history = [ {"role": "system", "content": "You are a helpful assistant that specializes in technical and general queries."}

import openai
from typing import List, Dict


def set_system_role(role_description: str, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Set the role of the assistant by modifying the system message.
    """
    conversation_history[0] = {"role": "system", "content": role_description}
    return conversation_history


def add_message_to_history(role: str, content: str, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Add a message to the conversation history.
    """
    conversation_history.append({"role": role, "content": content})
    return conversation_history


def chat_with_gpt(prompt: str, conversation_history: List[Dict[str, str]], model: str = "gpt-4", max_tokens: int = 4000, temperature: float = 0) -> (str, List[Dict[str, str]]):
    """
    Send a message to ChatGPT and get a response.

    Args:
        prompt: User's input message.
        conversation_history: List of messages in the conversation history.
        model: Model to use (e.g., gpt-4, gpt-3.5-turbo).
        max_tokens: Maximum tokens for the response.
        temperature: Creativity level of the response.

    Returns:
        Tuple containing:
        - ChatGPT's response as a string.
        - Updated conversation history.
    """
    # Add the user's message to the conversation history
    conversation_history.append({"role": "user", "content": prompt})

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)

    # Use the updated API to send the conversation to the model
    response = client.chat.completions.create(
        model=model,
        messages=conversation_history,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Extract the assistant's response
    assistant_response = response.choices[0].message.content

    # Add the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Return the response and updated conversation history
    return assistant_response, conversation_history


def clear_conversation_history() -> List[Dict[str, str]]:
    """
    Clear the conversation history to start fresh.

    Returns:
        A new conversation history list.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant that specializes in technical and general queries."}
    ]


def display_conversation_history(conversation_history: List[Dict[str, str]]) -> None:
    """
    Display the current conversation history.
    """
    for message in conversation_history:
        print(f"{message['role'].capitalize()}: {message['content']}\n")
