import os
import json
import pandas as pd
import time
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import re
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain.schema import LLMResult
import json

def strict_evaluator(question: str, ground_truth: str, answer: str):
    """
    Evaluates the student's answer against the ground truth strictly.

    Args:
        question (str): The question being asked.
        ground_truth (str): The correct answer.
        student_answer (str): The student's answer.

    Returns:
        int: Binary score, 1 if correct, 0 otherwise.
    """
    eval_prompt = PromptTemplate(
        input_variables=["question", "ground_truth", "answer"],
        template=(
            "You are a strict evaluator for answers. You will evaluate whether the student's answer strictly matches the ground truth." 
            "It is fine to have some leeway in terms of numerical rounding. i.e 101.1 as ground truth and the student answer is 101.2 should be considered correct. "
            "It is also fine to have some leeway in terms of numerical formatting. i.e. 123443 as ground truth and the students answer is 123,443 should be considered correct."
            "It is okay if the order of string values is not identical. e.g. ['IDX54421', 'IDX4223', 'IDA7786'] as ground truth and the students answer is ['IDA7786','IDX4223','IDX54421',] should be considered correct."
            "Provide a binary score (1 or 0) based on correctness.\n\n"
            "Question: {question}\n"
            "Ground Truth: {ground_truth}\n"
            "Student's Answer: {answer}\n\n"
            "Does the student's answer strictly match the ground truth? If yes, respond with 1. If no, respond with 0."
        ),
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    eval_chain = LLMChain(llm=llm, prompt=eval_prompt)

    result: LLMResult = eval_chain.run({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
    })

    match = re.search(r'\b(0|1)\b', result.strip())
    if match:
        return int(match.group(1))
    else:
        return 0  # Default to 0 if no valid score is found

def evaluate_json(input_file: str, output_file: str):
    """
    Evaluates answers from a JSON file and appends scores.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    for item in data:
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        student_answer = item.get("result", "")
        score = strict_evaluator(question, ground_truth, student_answer)
        item["score"] = score

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


class DataFrameAgentProcessor:
    def __init__(self, model_type: str, questions_path: str, model: str):
        """
        Initialize the DataFrameAgentProcessor.

        :param model_type: Specify the model type ('openai' or 'anthropic').
        :param questions_path: Path to the JSON file containing questions.
        """
        self.model_type = model_type
        self.questions_path = questions_path
        self.model_name = model

        if model_type.lower() == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
            self.model = ChatOpenAI(
                model=model, api_key=api_key, temperature=0.0
            )
        elif model_type.lower() == 'anthropic':
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
            self.model = ChatAnthropic(
                model=model, api_key=api_key, temperature=0.0
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
        cwd = os.getcwd()

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

            dataframes = [] 
            if os.path.isdir(dataset_path):
                for file_name in os.listdir(dataset_path):
                    file_path = os.path.join(dataset_path, file_name)
                    if file_name.endswith('.csv'):
                        try:
                            df = pd.read_csv(file_path)
                            dataframes.append(df)
                        except Exception as e:
                            data[i]['result'] = f"Error loading dataset from '{file_path}': {e}"
                            continue
            else:
                try:
                    df = pd.read_csv(dataset_path)
                    dataframes.append(df)
                except Exception as e:
                    data[i]['result'] = f"Error loading dataset: {e}"
                    continue

            if not dataframes:
                data[i]['result'] = "No valid datasets found to process."
                continue
            
            dataframes = dataframes if len(dataframes) >= 2 else dataframes[0]

            try:
                agent = create_pandas_dataframe_agent(
                    self.model,
                    dataframes,
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
        output_file = f"{base_name}_{self.model_type}_{self.model_name}_with_score.json"

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
                # time.sleep(50)



questions_folder = "./questions/" 
output_folder = "./results_folder/"

processor = DataFrameAgentProcessor(
    model_type="openai",
    questions_path="",
    model="gpt-4"          
)

processor.process_questions_folder(
    questions_folder=questions_folder,
    output_folder=output_folder
)