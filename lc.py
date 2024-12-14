import os
import json
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from evaluator.evaluator import evaluate_json

class DataFrameAgentProcessor:
    def __init__(self, model_type: str, questions_path: str):
        """
        Initialize the DataFrameAgentProcessor.

        :param model_type: Specify the model type ('openai' or 'anthropic').
        :param questions_path: Path to the JSON file containing questions.
        """
        self.model_type = model_type
        self.questions_path = questions_path

        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Questions file not found at: {questions_path}")

        if model_type.lower() == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
            self.model = ChatOpenAI(
                model="gpt-4o", api_key=api_key, temperature=0.0
            )
        elif model_type.lower() == 'anthropic':
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
            self.model = ChatAnthropic(
                model="claude-3-5-sonnet-latest", api_key=api_key, temperature=0.0
            )
        else:
            raise ValueError("Invalid model_type. Choose 'openai' or 'anthropic'.")

    def process_questions(self, output_path: str):
        """
        Process the questions and append results to the JSON file.

        :param output_path: Path to save the output JSON with results.
        """
        try:
            with open(self.questions_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading questions file: {e}")

        results = []
        cwd = os.getcwd()  # Get the current working directory

        for i, question_data in enumerate(data):
            dataset_path = question_data.get('table_path')
            if not dataset_path:
                data[i]['result'] = "Table path is invalid or missing."
                continue

            if not os.path.isabs(dataset_path):
                dataset_path = os.path.abspath(os.path.join(cwd, dataset_path))

            if not os.path.exists(dataset_path):
                data[i]['result'] = f"Table path '{dataset_path}' does not exist."
                continue

            dataframes = []  # List to hold DataFrames
            if os.path.isdir(dataset_path):  # Check if the path is a directory
                for file_name in os.listdir(dataset_path):
                    file_path = os.path.join(dataset_path, file_name)
                    if file_name.endswith('.csv'):  # Process only CSV files
                        try:
                            df = pd.read_csv(file_path)
                            dataframes.append(df)
                        except Exception as e:
                            data[i]['result'] = f"Error loading dataset from '{file_path}': {e}"
                            continue
            else:  # Single file path
                try:
                    df = pd.read_csv(dataset_path)
                    dataframes.append(df)
                except Exception as e:
                    data[i]['result'] = f"Error loading dataset: {e}"
                    continue

            if not dataframes:
                data[i]['result'] = "No valid datasets found to process."
                continue

            try:
                # Combine dataframes if needed or send as a list
                agent = create_pandas_dataframe_agent(
                    self.model,
                    dataframes,  # Pass list of DataFrames
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    system_prompt=(
                        "You are an expert data analyst. Answer the user's questions accurately based on the provided DataFrames. "
                        "Provide concise, relevant, and precise information derived from the data."
                    )
                )
            except Exception as e:
                data[i]['result'] = f"Error creating agent: {e}"
                continue

            question = question_data['question']
            try:
                result = agent.invoke(question)['output']
                data[i]['result'] = result
                results.append(result)
            except Exception as e:
                data[i]['result'] = str(e)

        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            raise IOError(f"Error saving results to output file: {e}")

        print(f"Processed results saved to '{output_path}'")

        self.run_evaluator(output_path)

    def run_evaluator(self, input_file: str):
        """
        Evaluate answers in the JSON file using strict evaluation.

        :param input_file: Path to the JSON file containing processed questions and answers.
        """
        base_name, _ = os.path.splitext(input_file)
        output_file = f"{base_name}_with_score.json"

        evaluate_json(input_file, output_file)
        print(f"Evaluation scores saved to '{output_file}'")


    def process_questions_folder(self, questions_folder: str, output_folder: str):
        """
        Process all JSON files in a folder and save results to corresponding output files.

        :param questions_folder: Path to the folder containing questions JSON files.
        :param output_folder: Path to the folder to save output JSON files.
        """
        if not os.path.exists(questions_folder):
            raise FileNotFoundError(f"Questions folder not found at: {questions_folder}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in os.listdir(questions_folder):
            if file_name.endswith('.json'):
                questions_path = os.path.join(questions_folder, file_name)
                output_path = os.path.join(output_folder, file_name)
                print(f"Processing file: {questions_path}")

                try:
                    self.questions_path = questions_path
                    self.process_questions(output_path)
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")



processor = DataFrameAgentProcessor(
    model_type="anthropic",
    questions_path="./questions/information_retrieval_2_crm.json"
)
processor.process_questions(output_path="./questions/CreditCard_with_results.json")
# processor.process_questions_folder(
#     questions_folder="./questions_folder",
#     output_folder="./results_folder"
# )
