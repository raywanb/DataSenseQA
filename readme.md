## Questions

### Question Arrays

The questions for each dataset are formatted 
in a JSON array with each array stored in an
individual file. 
#### Example: Question Array
```
[{"question_1"}, ... , {question_n}]
```
Each question array focuses on one question 
category (_data curation, information retrieval or statistics_). Subtypes for all question catetegories can be found in
the [question overview sheet](https://docs.google.com/spreadsheets/d/1IAQKPhE_R-w3SKLvBxDuQ01eqiVHOlhHdgtxtKnwJM8/edit?usp=sharing). 
Each question array contains **at least one question of each subtype in either difficulty**.

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
(_data curation, content retrieval or statistics_) and 
- **`subtype`** contains
the category subtypes that can be found in the [question overview sheet](https://docs.google.com/spreadsheets/d/1IAQKPhE_R-w3SKLvBxDuQ01eqiVHOlhHdgtxtKnwJM8/edit?usp=sharing).

#### Example: Question in JSON

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


