from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

anthropic_model = ChatAnthropic(model='claude-3-opus-20240229', api_key="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")
openai_model = ChatOpenAI(model="gpt-4o")

import pandas as pd
import json

with open('./questions/CreditCard.json', 'r') as f:
    data = json.load(f)


results = []

df = pd.read_csv("./datasets/demographics/users_data.csv")

agent = create_pandas_dataframe_agent(anthropic_model, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

for i in range(len(data)):
    # print(data[i])

    # table_path = data[i]['table_path']
    question = data[i]['question']
    # print(question)
    result = agent.invoke(question)

    # print(result)
    # print(data[i]['ground_truth'])
    results.append(result)

print(results)


   









