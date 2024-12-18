# DataSense: Evaluating LLM Agents on Tabular Data for Curation, Retrieval, and Statistical Analysis

##### Benchmark Track | 4 Units | Group TBD

 by Felix Krumme, Ray Wan, Dimple Amitha Garuadapuri, Manan Dhanuka

12/18/2024

-------------

## 1 Components & Files
##
### 1.1 `data_analysis` Directory

Contains further data analysis scripts

### 1.2 `datasets` Directory

The datasets folder contains all raw datasets in CSV format.

### 1.3 `questions` Directory

The questions directory contains all question collection sets in a JSON format.

### 1.4 `results_folder` Directory

The question sets with the scoring from the evaluator are stored here for every tested model.

### 1.5 `langchain_agent.py` Script

This script contains the code for the test_agent and the evaluator.

### 1.6 `langchain_run_script.ipynb` Jupyter Notebook

This notebook contains the code to run the benchmark on all questions for a specific LLM

### 1.7 `result_analysis.py` Script

This script contains the code to create general information for the benchmarks results

### 1.8 `visualize.ipynb` Jupyter Notebook

This notebook contains the code to create graphs to visualize the benchmark results

---------------

## 2 Setup
##
To integrate and use the LangChain agent with Large Language Models (LLMs), you need to provide the necessary API keys for the LLM services.

### 2.1 Create a `.env` File

Create a `.env` file in the root of the project directory (if one doesn't exist already)

### 2.2 Add API keys 
Add your API keys for the respective LLMs into the .env file. The format should be as follows:

    OPENAI_API_KEY="your_api_key_here"
    ANTHROPIC_API_KEY="your_api_key_here"
    GEMINI_API_KEY="your_api_key_here"
    MISTRAL_API_KEY="your_api_key_here"

If you do **NOT** have all API keys yet, get them here:

- [OPENAI](https://platform.openai.com/docs/overview)
- [ANTHROPIC](https://console.anthropic.com/login?selectAccount=true&returnTo=%2Fsettings%2Fkeys%3F)
- [GEMINI](https://ai.google.dev/gemini-api/docs/api-key)
- [MISTRAL](https://auth.mistral.ai/ui/login?flow=1be720ed-8a74-4e25-8034-4c837cc6e28e)

### 2.3 Install Dependencies from `requirements.txt`

Ensure that you have the necessary Python packages installed for working with environment variables and the LangChain agent

#### 2.3.1 Locate the `requirements.txt` file 

Locate the `reuirements.txt` file within this repository to see the required packages and versions.

#### 2.3.2 Install requirements

To install the requirements use the following command

    pip install -r requirements.txt

If these steps did **NOT** work, you have to install the dependencies manually, by going throught the files and installing all packages that throw errors.

---

## 3 Benchmark

### 3.1 Run Individual Question Sets

#### 3.1.1 Navigate to the `langchain_agent.py` file

This file is where the test agent execution and the evaluation happen. Open the file

#### 3.1.1 Edit the `main` method of `langchain_agent.py`

Within the `langchain_agent.py` file navigate to the `main` function at the bottom of the script.
It should look something like this:

    if __name__ == "__main__":
    
    # DO NOT EDIT
    questions_folder = "./questions/" 
    output_folder = "./results_folder/"

    # Edit this to the model you want to test
    processor = DataFrameAgentProcessor(
        model_type="anthropic",
        questions_path="",
        model="claude-3-5-haiku-latest"          
    )

    # Edit this path to the question collection you want to execute
    processor.process_questions_list(
        ["./questions/statistics_4_hedge_fund_questions.json"],
        output_folder
    )

##### Edit the model you want to test 

- Change the `model_type` to the model you want to test (`openai`,`gemini`, `mistral` or `anthropic` )
- Change the `model` to the specific model you want to test

##### Edit the path to the question set you want to test

Edit the path file and set it to the question set you want to evaluate. All question sets are contained within the question directory.

#### 3.1.3 Run the `main` method of the `langchain_agent.py`file.

After editing the `main` method to the model and question collection you want to test, run the main method.

#### 3.1.4 Check the output in the `results_folder` directory

The results of the model you just ran should appear in the `result_folder` directory.

### 3.2 Run All Questions

#### 3.2.1 Navigate to to the `langchain_run_script.ipynb` file

This Jupyter notebook can be executed to run all questions at once

#### 3.2.2 Change the model parameters 

Change the model parameters to the model you want to run

#### 3.2.3 Execute Script

Execute the script and find all questions answered in the `results_folder` directory.

---
## 4. Result Analysis
We provide a script to calculate key metrics to evaluate the benchmark results

### 4.1 Navigate to `result_analysis.py`

This file contains the code to create the comprehensive model comparison.

### 4.1 Run `main` to create the report

Navigate to the `main` method and run it to generate the report `comprehensive_model_comparison.txt`

---
## 5. Result Visualization

We include a Jupyter notebook to create different graphs that help with visualizing the benchmark results.

### 5.1 Navigate to `visualize.ipynb` 

This file contains the code to visualize the benchmark data

### 5.2 Run `visualize.ipynb` to create graphs

Run the notebook to create graphs that visualize the benchmark results

---

This concludes the DataSense
Benchmark by Felix Krumme, Ray Wan, Dimple Amitha Garuadapuri, Manan Dhanuka
