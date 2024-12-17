# DataSense: Evaluating LLM Agents on Tabular Data for Curation, Retrieval, and Statistical Analysis

##### Benchmark Track | 4 Units | Group TBD

Felix Krumme, Ray Wan, Dimple Amitha Garuadapuri, Manan Dhanuka

-------------

## 1 Components & Files
##
### 1.1 `[deprecated]` Directory

Contains files that are no longer relevant.

### 1.2 `datasets` Directory

The datasets folder contains all raw datasets in CSV format.

### 1.3 `evaluator` Directory

The evaluator directory contains the evaluator scripts.

### 1.4 `questions` Directory

The questions directory contains all question collection sets in a JSON format.

### 1.5 `results_folder` Directory

The questionsets with the scoring from the evaluator are stored here for every tested model.

### 1.6 `test_agents` Directory

The scripts for the tested agent framework are contained in this directory

---------------

## 2 Setup
##
To integrate and use the LangChain agent with Large Language Models (LLMs), you need to provide the necessary API keys for the LLM services.

### 2.1 Create a `.env` File

Create a `.env` file in the root of the project directory (if one doesn't exist already)

### 2.2 Add API keys 
Add your API keys for the respective LLMs into the .env file. The format should be as follows

    LLM_API_KEY_MODEL_1="your_api_key_here"
    LLM_API_KEY_MODEL_2="your_api_key_here"

### 2.3 Install Dependencies from `requirements.txt`

Ensure that you have the necessary Python packages installed for working with environment variables and the LangChain agent

#### 2.3.1 Locate the `requirements.txt` file 

Locate the `reuirements.txt` file within this repository to see the required packages and versions.

#### 2.3.2 Install requirements

To install the requirements use the following command

    pip install -r requirements.txt

---

## 3 Running the Benchmark

