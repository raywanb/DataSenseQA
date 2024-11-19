## Data

### Data Collection
All datasets that were used to create questions to test the individual agents on are open source. 
The domains of the dataset are the following:

- **Business:** financial data and ERP-System tables
  - h
- **Research:** demographic, genomic and survey datasets
  - Crimes reported in Chicago in 2019 (`Crimes_-_2019.csv`)
  - Covid-19 infection case data for various countries (`COVID19.csv`)
- **Application:** datasets of varíous different applications
  - h


## Question
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

