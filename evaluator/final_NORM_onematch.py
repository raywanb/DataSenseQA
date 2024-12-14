import json
import re
import argparse
from typing import List, Dict, Any

class EnhancedEvaluator:
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer using WTQ's NORM method:
        - Remove articles, punctuation
        - Convert to lowercase
        - Standardize numbers and units
        - Handle special cases like dates
        """
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove articles and extra whitespace
        answer = re.sub(r'\b(a|an|the)\b', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Standardize numbers
        answer = re.sub(r'(\d+),(\d+)', r'\1\2', answer)  # Remove commas in numbers
        answer = re.sub(r'(\d+)k\b', lambda m: str(int(m.group(1)) * 1000), answer)  # Convert 'k' to thousands
        
        # Standardize units
        unit_mappings = {
            'percent': '%',
            'percentage': '%',
            'dollars': '$',
            'dollar': '$',
            'usd': '$'
        }
        for word, symbol in unit_mappings.items():
            answer = answer.replace(word, symbol)
        
        # Standardize date formats
        month_mappings = {
            'january': '1', 'february': '2', 'march': '3', 'april': '4',
            'may': '5', 'june': '6', 'july': '7', 'august': '8',
            'september': '9', 'october': '10', 'november': '11', 'december': '12'
        }
        for month, num in month_mappings.items():
            answer = answer.replace(month, num)
        
        # Remove punctuation except necessary symbols
        answer = re.sub(r'[^\w\s$%.-]', '', answer)
        
        return answer.strip()

    def evaluate_answers(self, predictions_path: str, ground_truths_path: str) -> Dict[str, Any]:
        """
        Evaluate answer predictions against ground truths using enhanced normalization
        and detailed metrics breakdown.
        """
        # Load predictions and ground truths
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        with open(ground_truths_path, 'r', encoding='utf-8') as f:
            ground_truths = json.load(f)

        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")

        results = []
        total_correct = 0
        total_weighted_score = 0

        for pred, truth in zip(predictions, ground_truths):
            # Extract answer and ground truth
            pred_answer = pred.get('answer', '')
            truth_answer = truth.get('ground_truth', '')
            
            # Normalize both strings using WTQ NORM method
            norm_pred = self.normalize_answer(pred_answer)
            norm_truth = self.normalize_answer(truth_answer)
            
            # Check if normalized strings match
            is_correct = norm_pred == norm_truth
            
            # Calculate points based on difficulty
            difficulty = truth.get('difficulty', 'easy')
            points = {'hard': 3, 'medium': 2, 'easy': 1}.get(difficulty, 1)
            
            # Store detailed result
            result = {
                'question': truth.get('question', ''),
                'prediction': pred_answer,
                'normalized_prediction': norm_pred,
                'ground_truth': truth_answer,
                'normalized_ground_truth': norm_truth,
                'is_correct': is_correct,
                'difficulty': difficulty,
                'type': truth.get('type', ''),
                'subtype': truth.get('subtype', ''),
                'points': points,
                'weighted_score': points if is_correct else 0
            }
            results.append(result)
            
            if is_correct:
                total_correct += 1
                total_weighted_score += points

        # Calculate overall metrics
        total_questions = len(predictions)
        total_possible_points = sum(r['points'] for r in results)
        
        # Calculate difficulty breakdown
        difficulty_metrics = self._calculate_difficulty_breakdown(results)
        
        # Calculate type and subtype metrics
        type_metrics = self._calculate_type_metrics(results)

        return {
            'overall_metrics': {
                'exact_match_accuracy': total_correct / total_questions,
                'weighted_score': total_weighted_score / total_possible_points,
                'total_questions': total_questions,
                'total_correct': total_correct,
                'total_points': total_weighted_score,
                'possible_points': total_possible_points
            },
            'difficulty_metrics': difficulty_metrics,
            'type_metrics': type_metrics,
            'detailed_results': results
        }

    def _calculate_difficulty_breakdown(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate accuracy metrics broken down by difficulty level"""
        difficulty_metrics = {}
        
        for diff in ['easy', 'medium', 'hard']:
            diff_results = [r for r in results if r['difficulty'] == diff]
            if diff_results:
                correct = sum(1 for r in diff_results if r['is_correct'])
                total = len(diff_results)
                diff_points = sum(r['weighted_score'] for r in diff_results)
                possible_points = sum(r['points'] for r in diff_results)
                
                difficulty_metrics[diff] = {
                    'accuracy': correct / total,
                    'weighted_score': diff_points / possible_points,
                    'total': total,
                    'correct': correct,
                    'points_earned': diff_points,
                    'points_possible': possible_points
                }
        
        return difficulty_metrics

    def _calculate_type_metrics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate accuracy metrics broken down by type and subtype"""
        type_metrics = {}
        
        for result in results:
            result_type = result['type']
            result_subtype = result['subtype']
            
            if result_type not in type_metrics:
                type_metrics[result_type] = {
                    'total': 0, 
                    'correct': 0, 
                    'points_earned': 0,
                    'points_possible': 0,
                    'subtypes': {}
                }
                
            type_metrics[result_type]['total'] += 1
            type_metrics[result_type]['points_possible'] += result['points']
            
            if result['is_correct']:
                type_metrics[result_type]['correct'] += 1
                type_metrics[result_type]['points_earned'] += result['weighted_score']
                
            if result_subtype:
                if result_subtype not in type_metrics[result_type]['subtypes']:
                    type_metrics[result_type]['subtypes'][result_subtype] = {
                        'total': 0,
                        'correct': 0,
                        'points_earned': 0,
                        'points_possible': 0
                    }
                    
                subtype_metrics = type_metrics[result_type]['subtypes'][result_subtype]
                subtype_metrics['total'] += 1
                subtype_metrics['points_possible'] += result['points']
                
                if result['is_correct']:
                    subtype_metrics['correct'] += 1
                    subtype_metrics['points_earned'] += result['weighted_score']
        
        # Calculate accuracies and weighted scores
        for type_data in type_metrics.values():
            type_data['accuracy'] = type_data['correct'] / type_data['total']
            type_data['weighted_score'] = (type_data['points_earned'] / 
                                         type_data['points_possible'] if type_data['points_possible'] > 0 else 0)
            
            for subtype_data in type_data['subtypes'].values():
                subtype_data['accuracy'] = subtype_data['correct'] / subtype_data['total']
                subtype_data['weighted_score'] = (subtype_data['points_earned'] / 
                                                subtype_data['points_possible'] if subtype_data['points_possible'] > 0 else 0)
        
        return type_metrics

def main():
    """
    Main function to run evaluation from command line
    """
    parser = argparse.ArgumentParser(description='Evaluate answer predictions using enhanced normalization')
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON file')
    parser.add_argument('--ground-truths', required=True, help='Path to ground truths JSON file')
    parser.add_argument('--output', help='Path to save evaluation results JSON')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = EnhancedEvaluator()
    results = evaluator.evaluate_answers(args.predictions, args.ground_truths)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Save results if output path provided
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()