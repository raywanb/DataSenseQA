from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import math

def extract_question_metadata(input_data: dict) -> tuple[str, str, dict]:
    """
    Extract metadata from question input data, returning ground truth and result as 
    separate strings and remaining metadata as a dictionary.
    
    Args:
        input_data (dict): Input JSON containing question data
        
    Returns:
        tuple: (ground_truth: str, result: str, metadata: dict)
            - ground_truth: The ground truth string
            - result: The result output string
            - metadata: Dictionary containing difficulty, type and subtype
    """
    ground_truth = input_data.get('ground_truth', '')
    result = input_data.get('result', {}).get('output', '') if input_data.get('result') else ''
    
    metadata = {
        'difficulty': input_data.get('difficulty', ''),
        'type': input_data.get('type', ''),
        'subtype': input_data.get('subtype', '')
    }
    
    return ground_truth, result, metadata





def calculate_accuracy_metrics(results: List[tuple[str, str, dict]]) -> Dict[str, Dict]:
    """
    Calculate accuracy metrics broken down by type and subtype using ground truth 
    and result comparisons.
    
    Args:
        results: List of tuples containing (ground_truth, result, metadata)
        
    Returns:
        Dictionary containing accuracy metrics by type and subtype
    """
    type_metrics = {}
    
    for ground_truth, result, metadata in results:
        result_type = metadata['type']
        result_subtype = metadata['subtype']
        
        # Initialize type metrics if not exists
        if result_type not in type_metrics:
            type_metrics[result_type] = {
                'total': 0,
                'correct': 0,
                'subtypes': {}
            }
        
        # Update type counts
        type_metrics[result_type]['total'] += 1
        
        # Compare ground truth with result
        # For this example, using exact string matching
        # Could be replaced with more sophisticated comparison
        is_correct = ground_truth.strip() == result.strip()
        if is_correct:
            type_metrics[result_type]['correct'] += 1
            
        # Handle subtype metrics
        if result_subtype:
            if result_subtype not in type_metrics[result_type]['subtypes']:
                type_metrics[result_type]['subtypes'][result_subtype] = {
                    'total': 0,
                    'correct': 0
                }
                
            subtype_metrics = type_metrics[result_type]['subtypes'][result_subtype]
            subtype_metrics['total'] += 1
            if is_correct:
                subtype_metrics['correct'] += 1
    
    # Calculate accuracies
    for type_data in type_metrics.values():
        type_data['accuracy'] = type_data['correct'] / type_data['total'] if type_data['total'] > 0 else 0
        
        for subtype_data in type_data['subtypes'].values():
            subtype_data['accuracy'] = subtype_data['correct'] / subtype_data['total'] if subtype_data['total'] > 0 else 0
    
    return type_metrics


# metrics = calculate_accuracy_metrics(sample_results)


# Do not modify the signature of the "main" function.
def main(user_query: str):
    entrypoint_agent_system_message = "You are an AI assistant to evaluate the performance of a data analyst." # TODO

    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    # the main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                        system_message=entrypoint_agent_system_message, 
                                        llm_config=llm_config)
    entrypoint_agent.register_for_llm(name="extract_question_metadata", description="extracts the required data from the input string: ground_truth, result, metadata = extract_question_metadata(input_data)")(extract_question_metadata)
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(extract_question_metadata)
   

    accuracy_agent = ConversableAgent("accuracy_agent", system_message="You are an agent that determines whether the result and the groundtruth are equivalent or highlights the differences", llm_config=llm_config)
    
    evaluation_agent_1 = ConversableAgent("evaluation_agent_1", system_message="You are an agent that analyzes how wrong the result is from the groundtruth. provide a score from 1-10 where 1 means it is entirely wrong, and 10 means it is entirely correct." , llm_config=llm_config)

    evaluation_agent_2 = ConversableAgent("evaluation_agent_1", system_message="You are an agent that analyzes score from 1-10 where 1 means it is 10 means icorrect.", llm_config=llm_config)

    evaluation_agent_1.register_for_llm(name="calculate_accuracy_metrics", description="calculates the final restaurant score.")(calculate_accuracy_metrics)
    evaluation_agent_1.register_for_execution(name="calculate_accuracy_metrics")(calculate_accuracy_metrics)
    evaluation_agent_2.register_for_llm(name="calculate_accuracy_metrics", description="calculates the final restaurant score.")(calculate_accuracy_metrics)
    evaluation_agent_2.register_for_execution(name="calculate_accuracy_metrics")(calculate_accuracy_metrics)
    

    result = entrypoint_agent.initiate_chats([{
            "recipient": accuracy_agent,
            "max_turns": 2, 
            "summary_method": "last_msg",
            "message": f'is the result accurate?'
        }, 
        {
            "recipient": evaluation_agent_1,  
            "max_turns": 1, 
            "summary_method": "last_msg",
            "message": f'how accurate is the result?'
        },
        {
            "recipient": evaluation_agent_2,  
            "max_turns": 2, 
            "summary_method": "last_msg",
            "message": f'what is the overall performance of the data analyst?'
        }])
   
# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
