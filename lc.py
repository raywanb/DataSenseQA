from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import pandas as pd
import json


anthropic_model = ChatAnthropic(model='claude-3-5-sonnet-latest', api_key="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", temperature=0.0)
openai_model = ChatOpenAI(model="gpt-4o")

with open('./questions/CreditCard.json', 'r') as f:
    data = json.load(f)

df = pd.read_csv("./datasets/demographics/users_data.csv")

system_prompt = (
    "You are an expert data analyst. Answer the user's questions accurately based on the provided DataFrame. "
    "Provide concise, relevant, and precise information derived from the data."
)

agent = create_pandas_dataframe_agent(
    anthropic_model,
    df,
    verbose=True,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    system_prompt=system_prompt
)

results = []
for i in range(len(data)):
    question = data[i]['question']
    try:
        result = agent.invoke(question)
        data[i]['result'] = result
        results.append(result)
    except Exception as e:
        data[i]['result'] = str(e)

with open('./questions/CreditCard_with_results.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Processed results saved to './questions/CreditCard_with_results.json'")


   









