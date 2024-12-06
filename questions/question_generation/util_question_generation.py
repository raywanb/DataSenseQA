import pandas as pd
import json
import openai
import sys
import os

from questions.question_generation.chatGPT_interactions import set_system_role

sys.path.append("../../test_agents/chatGPT")

import chatGPT_interactions as gpt
from importlib import reload
reload(gpt)

class FileAnalysis:
    @staticmethod
    def get_table_info(file_path, file_name="csv_file.csv"):
        """
        Summarizes the contents of a CSV file.

        Parameters:
            file_path (str): Path to the CSV file.
            file_name (str): Name of the file for display (default: "CSV File").

        Returns:
            dict: A summary dictionary containing the number of columns, rows, total fields, and empty fields.
        """
        try:
            data = pd.read_csv(file_path)

            num_columns = data.shape[1]
            num_rows = data.shape[0]
            total_fields = num_columns * num_rows
            empty_fields = data.isnull().sum().sum()

            return {
                "file_name": file_name,
                "num_columns": num_columns,
                "num_rows": num_rows,
                "total_fields": total_fields,
                "empty_fields": empty_fields
            }
        except FileNotFoundError:
            print(f"Error: File not found at path: {file_path}")
            return None

    @staticmethod
    def get_column_info(file_path, file_name="CSV File", n_unique=5):
        """
        Summarizes the columns of a CSV file, including unique values, number of unique values,
        number of missing values, and additional notes if applicable.

        Parameters:
            file_path (str): Path to the CSV file.
            file_name (str): Name of the file for display (default: "CSV File").
            n_unique (int): Number of unique values (default: 5).

        Returns:
            str: A JSON-formatted string summarizing the columns of the dataset.
        """
        try:
            data = pd.read_csv(file_path)
            summary = {}
            for column in data.columns:
                unique_values = data[column].dropna().unique()
                num_unique = len(unique_values)
                missing_values = data[column].isnull().sum()

                column_summary = {
                    "unique_values": [str(value) for value in unique_values[:n_unique]] if num_unique > n_unique else [
                        str(value) for
                        value in
                        unique_values],
                    "number_of_unique_values": int(num_unique),  # Convert to native Python int
                    "number_of_missing_values": int(missing_values)  # Convert to native Python int
                }
                if num_unique > n_unique:
                    column_summary["note"] = "More unique values contained than printed!"
                summary[column] = column_summary

            json_summary = json.dumps(summary, indent=4)

            return json_summary
        except FileNotFoundError:
            print(f"Error: File not found at path: {file_path}")
            return None

    import pandas as pd
    import json

    @staticmethod
    def get_table_top_rows(file_path, file_name="CSV File", n_rows=10):
        """
        Reads a CSV file and returns the top N rows in JSON format.

        Parameters:
            file_path (str): Path to the CSV file.
            file_name (str): Name of the file for display (default: "CSV File").
            n_rows (int): Number of rows to select from the top of the dataset (default: 10).

        Returns:
            str: A JSON-formatted string of the top N rows.
        """
        try:
            df = pd.read_csv(file_path)
            top_rows = df.head(n_rows)
            result_json = top_rows.to_json(orient='records', indent=4)

            return result_json

        except FileNotFoundError:
            print(f"Error: File not found at path: {file_path}")
            return None

    @staticmethod
    def load_json_file(file_path):
        """
        Load a JSON file from the given path and return its contents.

        Parameters:
        - file_path (str): The path to the JSON file.

        Returns:
        - dict or list: The contents of the JSON file as a Python dictionary or list.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: The file at path '{file_path}' was not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from the file at path '{file_path}'.")
            return None

    # Example usage:
    # json_data = load_json_file('path/to/your/file.json')
    # print(json_data)

    @staticmethod
    def get_table_description(prompt: str = ""):
        conversation_history = [{"role": "system", "content": "You are a helpful assistant that specializes in technical and general queries."}]
        
        role_description = ""
        set_system_role()
        return "answer"

