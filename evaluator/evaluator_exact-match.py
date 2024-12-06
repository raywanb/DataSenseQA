import json
import re

def normalize_string(s):
    """
    Normalize a string by:
    1. Converting to lowercase
    2. Removing extra whitespace
    3. Standardizing number formats
    4. Removing common currency symbols and commas in numbers
    """
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)  # Standardize whitespace
    s = s.replace('$', '').replace(',', '')  # Remove currency and number formatting
    return s

def normalize_numbers(text):
    """
    Find all numbers in text and standardize their format
    """
    # Convert percentage numbers
    text = re.sub(r'(\d+\.?\d*)%', lambda m: f"{float(m.group(1)):.1f}%", text)
    
    # Convert decimal numbers
    text = re.sub(r'(\d+)\.(\d+)', lambda m: f"{float(m.group(0)):.2f}", text)
    
    return text

def evaluate_answers(predictions, ground_truths):
    """
    Evaluate answer predictions against ground truths using a modified exact match approach
    that checks if the normalized ground truth is contained within the normalized prediction.
    
    Args:
        predictions (list): List of dictionaries containing answer predictions
        ground_truths (list): List of dictionaries containing ground truth data
        
    Returns:
        dict: Dictionary containing evaluation metrics and detailed results
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Number of predictions and ground truths must match")
    
    results = []
    correct = 0
    
    for pred, truth in zip(predictions, ground_truths):
        # Extract answer and ground truth
        pred_answer = pred.get('answer', '')
        truth_answer = truth.get('ground_truth', '')
        
        # Normalize both strings
        norm_pred = normalize_string(normalize_numbers(pred_answer))
        norm_truth = normalize_string(normalize_numbers(truth_answer))
        
        # Check if normalized ground truth is in normalized prediction
        is_correct = norm_truth in norm_pred
        
        # Store detailed result
        result = {
            'question': truth.get('question', ''),
            'prediction': pred_answer,
            'ground_truth': truth_answer,
            'is_correct': is_correct,
            'difficulty': truth.get('difficulty', ''),
            'type': truth.get('type', ''),
            'subtype': truth.get('subtype', '')
        }
        results.append(result)
        
        if is_correct:
            correct += 1
    
    # Calculate metrics
    accuracy = correct / len(predictions) if predictions else 0
    
    # Calculate accuracy by difficulty
    difficulty_metrics = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_results = [r for r in results if r['difficulty'] == diff]
        if diff_results:
            diff_correct = sum(1 for r in diff_results if r['is_correct'])
            diff_accuracy = diff_correct / len(diff_results)
            difficulty_metrics[diff] = {
                'accuracy': diff_accuracy,
                'total': len(diff_results),
                'correct': diff_correct
            }
    
    # Calculate accuracy by type and subtype
    type_metrics = {}
    for result in results:
        result_type = result['type']
        result_subtype = result['subtype']
        
        if result_type not in type_metrics:
            type_metrics[result_type] = {'total': 0, 'correct': 0, 'subtypes': {}}
            
        type_metrics[result_type]['total'] += 1
        if result['is_correct']:
            type_metrics[result_type]['correct'] += 1
            
        if result_subtype not in type_metrics[result_type]['subtypes']:
            type_metrics[result_type]['subtypes'][result_subtype] = {'total': 0, 'correct': 0}
            
        type_metrics[result_type]['subtypes'][result_subtype]['total'] += 1
        if result['is_correct']:
            type_metrics[result_type]['subtypes'][result_subtype]['correct'] += 1
    
    # Calculate accuracies for types and subtypes
    for type_name, type_data in type_metrics.items():
        type_data['accuracy'] = type_data['correct'] / type_data['total']
        for subtype_name, subtype_data in type_data['subtypes'].items():
            subtype_data['accuracy'] = subtype_data['correct'] / subtype_data['total']
    
    # Calculate weighted score
    total_possible_points = 0
    earned_points = 0
    
    for result in results:
        if result['difficulty'] == 'hard':
            total_possible_points += 3
            if result['is_correct']:
                earned_points += 3
        elif result['difficulty'] == 'medium':
            total_possible_points += 2
            if result['is_correct']:
                earned_points += 2
        else:  # easy
            total_possible_points += 1
            if result['is_correct']:
                earned_points += 1
    
    weighted_score = earned_points / total_possible_points if total_possible_points > 0 else 0
    
    return {
        'overall_accuracy': accuracy,
        'weighted_score': weighted_score,
        'weighted_score_details': {
            'earned_points': earned_points,
            'total_possible_points': total_possible_points,
            'points_by_difficulty': {
                'hard': {'points': 3, 'count': len([r for r in results if r['difficulty'] == 'hard'])},
                'medium': {'points': 2, 'count': len([r for r in results if r['difficulty'] == 'medium'])},
                'easy': {'points': 1, 'count': len([r for r in results if r['difficulty'] == 'easy'])}
            }
        },
        'total_examples': len(predictions),
        'total_correct': correct,
        'difficulty_metrics': difficulty_metrics,
        'type_metrics': type_metrics,
        'detailed_results': results
    }

# Example usage:
if __name__ == "__main__":
    # Sample predictions and ground truths
    predictions = [
        {"answer": "The total revenue for Won deals is $10,005,534"},
        {"answer": "GTX Basic appears most frequently in Won deals"}
    ]
    
    ground_truths = [
        {
            "question": "Calculate the total revenue generated by Won deals.",
            "ground_truth": "10,005,534",
            "difficulty": "easy",
            "type": "information retrieval",
            "subtype": "Multi-Column Analysis and Summarization"
        },
        {
            "question": "Identify the product that appears most frequently in 'Won' deals.",
            "ground_truth": "GTX Basic",
            "difficulty": "easy",
            "type": "information retrieval",
            "subtype": "Pattern and Trend Recognition"
        }
    ]
    
    # Run evaluation
    evaluation_results = evaluate_answers(predictions, ground_truths)
    print(json.dumps(evaluation_results, indent=2))