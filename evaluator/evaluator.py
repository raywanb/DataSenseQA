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
            "You are a strict evaluator for answers. You will evaluate whether the student's answer strictly matches the ground truth. "
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

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to save the output JSON file with scores.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    for item in data:
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        student_answer = item.get("result", "")
        score = strict_evaluator(question, ground_truth, student_answer)
        item["score"] = score

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
