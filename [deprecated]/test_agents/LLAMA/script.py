import os
import pandas as pd
from io import StringIO
import json
from llama_index import (
    GPTSimpleVectorIndex,
    Document,
    ServiceContext
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI


class LlamaIndexDataFrameAgent:
    def __init__(self, csv_input, openai_api_key=None, is_file=True):

        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY env variable or pass as argument.")

        # Normalize csv_input into a list for consistent handling
        if isinstance(csv_input, str):
            csv_input = [csv_input]
        self.csv_inputs = csv_input
        self.is_file = is_file

        # Initialize the LLM context
        self.service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0, openai_api_key=self.openai_api_key)
        )

        # Build the index from multiple CSVs
        self.index = self._create_index()

    def _load_single_csv(self, csv_source):
        """Load a single CSV into a DataFrame."""
        if self.is_file:
            df = pd.read_csv(csv_source)
        else:
            csv_file_like = StringIO(csv_source)
            df = pd.read_csv(csv_file_like)
        return df

    def _create_index(self):
        """
        Create a GPTSimpleVectorIndex from multiple CSV files/strings.
        """
        documents = []

        for csv_source in self.csv_inputs:
            df = self._load_single_csv(csv_source)
            data_str = df.to_csv(index=False)
            document = Document(text=data_str)
            documents.append(document)

        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)

        index = GPTSimpleVectorIndex(nodes, service_context=self.service_context)
        return index

    def answer_question(self, question: str, difficulty: str = "easy", ground_truth: str = None,
                        derivation_code: str = None, output_json_path: str = "output.json"):

        response = self.index.query(question)
        agent_answer = str(response)

        evaluation_result = None
        if ground_truth or derivation_code:
            evaluation_result = self.evaluate_answer(question, agent_answer, ground_truth, derivation_code)

        result_dict = {
            "question": question,
            "agent_answer": agent_answer,
            "evaluation": evaluation_result,
            "difficulty": difficulty
        }

        # Write the result to a JSON file
        with open(output_json_path, "w") as f:
            json.dump(result_dict, f, indent=4)

        return result_dict

    def evaluate_answer(self, question: str, agent_answer: str, ground_truth: str = None, derivation_code: str = None):

        computed_truth = None
        if derivation_code:
            local_vars = {}
            global_vars = {}
            exec(derivation_code, global_vars, local_vars)
            computed_truth = local_vars.get('derived_answer', None)

        expected_answer = computed_truth if computed_truth is not None else ground_truth

        if expected_answer is not None:
            is_match = agent_answer.strip() == expected_answer.strip()
        else:
            is_match = False

        return {
            "match": is_match,
            "expected_answer": expected_answer,
            "scoring": 1.0 if is_match else 0.0,
            "notes": "Exact match comparison used."
        }


# Example usage
if __name__ == "__main__":
    # Example with multiple CSVs:
    csv_paths = ["business_data.csv", "research_data.csv"]
    agent = LlamaIndexDataFrameAgent(csv_paths, openai_api_key="your_api_key", is_file=True)

    question = "What is the average value in all the combined data?"
    difficulty = "medium"
    ground_truth = "The average is 42."  # Example known correct answer
    result = agent.answer_question(question, difficulty=difficulty, ground_truth=ground_truth,
                                   output_json_path="result.json")

    print("Result Dictionary:", result)
    # The result is also saved in "result.json"