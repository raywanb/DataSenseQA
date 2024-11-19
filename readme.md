# Datasets

### Dataset Collection
All datasets that were used to create questions to test the individual agents on are available on open source such as [kaggle](kaggle.com/search). An overview
all considered datasets and their sources can be found in the [dataset sheet](https://docs.google.com/spreadsheets/d/1IAQKPhE_R-w3SKLvBxDuQ01eqiVHOlhHdgtxtKnwJM8/edit?usp=sharing). 


### Dataset Domains
We declare different domains to have suitable datasets for different question focusses, namely data curation, 
content retrieval and statistics. Every agent will be tested on each dataset as we want to evaluate their performance across 
all domains and question focuses. 

The domains are the following:

- **Business:** financial data and ERP-System tables
  - DATA
- **Research:** demographic, genomic and survey datasets
  - Crimes reported in Chicago in 2019 (`Crimes_-_2019.csv`)
  - Covid-19 infection case data for various countries (`COVID19.csv`)
- **Application:** datasets of varíous different applications
  - DATA

# Questions
### Question Scope
Questions describe the individual tasks that are
used to test different agents and LLMs respectively. 
We want to design our questions as such that they are not too trivial
for the agents to solve, as we ideally want to identify areas of improvement for each agent and LLM.

### Question Structure
Each dataset is equipped with one question collection.
A question collection for a dataset focuses on one of the question 
categories (_data curation, information retrieval or statistics_). Subtypes for all question catetegories can be found in
the [question overview sheet](https://docs.google.com/spreadsheets/d/1IAQKPhE_R-w3SKLvBxDuQ01eqiVHOlhHdgtxtKnwJM8/edit?usp=sharing). 
Each question collection hereby contains **at least one question of each subtype in either difficulty**.

## Question Format

### Question Arrays
The questions for each dataset are formatted 
in a JSON array with each array stored in an
individual file. 
#### [Example]: Question Array
```
[{"question_1"}, ... , {question_n}]
```

### Question Structure

Each question consisting of the following 
attributes: 
- **`question`** contains the question written 
in a normal sentence with a question mark, 
- **`ground_truth`** contains the correct answer to the question, 
- **`derivation`** contains the `pandas` python 
code to derive the ground_truth, 
- **`difficulty`** labels each question as either _easy, medium_ or _hard_
- **`type`** contains the question category 
(_data curation, content retrieval or statistics_), 
- **`subtype`** contains
the category subtypes that can be found in the [question overview sheet](https://docs.google.com/spreadsheets/d/1IAQKPhE_R-w3SKLvBxDuQ01eqiVHOlhHdgtxtKnwJM8/edit?usp=sharing) and
- `table_path` indicating which table the question refers to.

#### [Example]: Question in JSON
```json
{
"question" : "What is the mean of column Age?",
"ground_truth" : "1.57",
"derivation" : "df['Age'].mean()",
"difficulty" : "easy",
"type" : "statistics",
"subtype" : "descriptive statistics",
"table_path" : "datasets/research/Example.csv"
}
```

#### [Template]: Empty Question in JSON
```json
{
"question" : "",
"ground_truth" : "",
"derivation" : "",
"difficulty" : "",
"type" : "",
"subtype" : "",
"table_path" : ""
}
```
## Question Generation

### Manually Generated Questions
Each dataset is assigned with one question category (_data curation, content retrieval or statisitcs_) first. 
To optimally design the questions, each category has the before mentioned question subtypes. We use our own human understanding 
to formulate questions for the dataset, derive a ground truth and label the questions accordingly. 

### LLM Generated Questions

To make sure every question subtype is included to uncover all possible weaknesses of the tested agents, the hand crafted 
question collection is prompted to ChatGPT with the first ten rows of the dataset, as well as an adequate description of the dataset,
including data quality issues to generate more questions. As the derivation is always included in the question, it is possibler check if the ground truth is in fact true.  

### Question Valuation
First  all question collections are crossexamed by hand. To make sure that no issues with the question are overseen,the question collection including the first ten rows of the underlying dataset are prompted into ChatGPT-4 in three different instances
to make sure that all questions pass all acception criteria (_from: InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks_):


- **Suitableness:** Evaluates if the CSV file is suitable for data analysis.
- **Reasonableness:** Ensures the question, constraints, and format are natural and conflict-free.
- **Value:** Assesses the practical usefulness of the questions generated.
- **Restrictiveness:** Checks if constraints are strict enough to ensure unique answers.
- **Alignment:** Verifies that questions align with the data’s content, type, and range.
- **Correctness:** Confirms that question labels are accurate.



