import json
import pandas as pd

def load_questions_from_json(path):
    """
    Loads a JSON file from the specified path and extracts all questions into an array.

    Args:
        path (str): Path to the JSON file.

    Returns:
        list: An array of questions.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON file format is invalid.
        Exception: For any other unexpected errors.
    """
    try:
        # Load the JSON file
        with open(path, 'r') as file:
            data = json.load(file)

        # Extract questions
        questions = [entry.get("question", "") for entry in data]
        return questions

    except FileNotFoundError as e:
        print(f"Error: File not found at {path}")
        raise e
    except json.JSONDecodeError as e:
        print("Error: Failed to decode JSON. Please check the file format.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

def load_dataframe_from_csv(path):
    """
    Loads a CSV file from the specified path into a Pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file format is invalid.
        Exception: For any other unexpected errors.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(path)
        return df

    except FileNotFoundError as e:
        print(f"Error: File not found at {path}")
        raise e
    except pd.errors.EmptyDataError as e:
        print("Error: The CSV file is empty.")
        raise e
    except pd.errors.ParserError as e:
        print("Error: Failed to parse the CSV file. Please check the file format.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e
