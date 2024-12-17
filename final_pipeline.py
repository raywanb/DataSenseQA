import os
from pathlib import Path
from langchain_agent import DataFrameAgentProcessor

def process_question_files(input_folder, output_folder):

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files in the input folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    for json_file in json_files:
        input_path = os.path.join(input_folder, json_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(json_file)[0]}_with_results.json")
        
        print(f"Processing {json_file}...")
        processor = DataFrameAgentProcessor(
            model_type="anthropic",
            questions_path=input_path
        )
        processor.process_questions(output_path=output_path)
        print(f"Completed processing {json_file}")

if __name__ == "__main__":
    input_folder = "./questions_research"
    output_folder = "./questions_research/results"
    process_question_files(input_folder, output_folder)