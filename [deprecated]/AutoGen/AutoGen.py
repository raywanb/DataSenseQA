
from autogen import ConversableAgent
import sys
import util
from dotenv import load_dotenv
import os
from autogen import register_function


load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

def get_data_agent_system_message() -> str:
    return "I am the data point agent, responsible for understanding and analyzing tabular data.I have to answer the questions that I get as precise as possible and in the correct format."

def get_llm_config():
    """
    Creates and returns the LLM configuration dictionary.

    Args:
        api_key (str): The API key for accessing the LLM.

    Returns:
        dict: The LLM configuration dictionary.
    """
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "api_key": api_key
            }
        ]
    }
    return llm_config

# Do not modify the signature of the "main" function.
def main(user_query: str):

    # Set up the entry point agent's system message
    data_agent_system_message = get_data_agent_system_message()

    # Get the API key from the environment variable

    if not api_key:
        raise ValueError(
            "The OPENAI_API_KEY environment variable is not set or is invalid. Please check your .env file."
        )

    # Example LLM configuration for the entrypoint agent
    llm_config = get_llm_config()

    # The main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent(
        "data_agent",
        system_message=data_agent_system_message,
        llm_config=llm_config
    )

    register_function(
        fetch_restaurant_data,
        caller=datafetch_agent,  # The assistant agent can suggest calls to the calculator.
        executor=entrypoint_agent,  # The user proxy agent can execute the calculator calls.
        name="fetch_dataset_questions",  # By default, the function name is used as the tool name.
        description="Fetches the reviews for a specific restaurant.",  # A description of the tool.
    )

    # The entry point agent sends the user query to the data fetch agent
    entrypoint_prompt = f"The user asked: '{user_query}'. Please process the query."

    # Initiate the chat sequence between the entrypoint agent and the datafetch agent
    result = entrypoint_agent.initiate_chats([
        {
            "message": entrypoint_prompt,
            "recipient": datafetch_agent,
            "max_turns": 1,
            "summary_method": "last_msg"
        },

    ])

# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
