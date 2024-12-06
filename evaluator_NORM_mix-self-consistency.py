import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from statistics import mode

class WTQEvaluator:
    def __init__(self, k_samples: int = 5):
        """
        Initialize evaluator with number of samples for self-consistency
        
        Args:
            k_samples: Number of samples to generate for each question in Mix self-consistency
        """
        self.k_samples = k_samples

    def normalize_answer(self, answer: str) -> str:
        """
        NORM mechanism: Normalize answers following WTQ standards
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

    def check_answer_match(self, pred: str, truth: str) -> bool:
        """
        Check if normalized prediction matches normalized ground truth
        """
        norm_pred = self.normalize_answer(pred)
        norm_truth = self.normalize_answer(truth)
        return norm_pred == norm_truth

    def evaluate_mix_consistency(self, predictions: List[List[str]], ground_truth: str) -> Tuple[bool, float]:
        """
        Evaluate using Mix self-consistency mechanism
        
        Args:
            predictions: List of k different model predictions for the same question
            ground_truth: Ground truth answer
            
        Returns:
            Tuple of (is_correct, confidence_score)
        """
        # Normalize all predictions and ground truth
        norm_predictions = [self.normalize_answer(pred) for pred in predictions]
        norm_truth = self.normalize_answer(ground_truth)
        
        # Get most common prediction
        prediction_counts = Counter(norm_predictions)
        most_common_pred = prediction_counts.most_common(1)[0][0]
        
        # Calculate confidence score
        confidence_score = prediction_counts[most_common_pred] / len(predictions)
        
        # Check if most common prediction matches ground truth
        is_correct = (most_common_pred == norm_truth)
        
        return is_correct, confidence_score

    def evaluate_answers(self, predictions_path: str, ground_truths_path: str) -> Dict[str, Any]:
        """
        Main evaluation function combining NORM and Mix self-consistency
        """
        # Load predictions and ground truths
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        with open(ground_truths_path, 'r', encoding='utf-8') as f:
            ground_truths = json.load(f)
            
        results = []
        total_correct = 0
        total_weighted_score = 0
        confidence_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            # Get sample predictions (assuming predictions contains k samples for each question)
            sample_predictions = pred.get('sample_answers', [pred.get('answer')])
            
            # Apply Mix self-consistency if multiple samples available
            if len(sample_predictions) > 1:
                is_correct, confidence = self.evaluate_mix_consistency(
                    sample_predictions, 
                    truth['ground_truth']
                )
            else:
                # Use single prediction with NORM
                is_correct = self.check_answer_match(
                    sample_predictions[0], 
                    truth['ground_truth']
                )
                confidence = 1.0
                
            # Calculate points based on difficulty
            points = {
                'hard': 3,
                'medium': 2,
                'easy': 1
            }.get(truth.get('difficulty', 'easy'), 1)
            
            result = {
                'question': truth.get('question', ''),
                'ground_truth': truth['ground_truth'],
                'prediction': sample_predictions[0],
                'all_predictions': sample_predictions,
                'is_correct': is_correct,
                'confidence': confidence,
                'difficulty': truth.get('difficulty', 'easy'),
                'points': points,
                'weighted_score': points if is_correct else 0
            }
            
            results.append(result)
            if is_correct:
                total_correct += 1
                total_weighted_score += points
            confidence_scores.append(confidence)
            
        # Calculate overall metrics
        total_possible_points = sum(r['points'] for r in results)
        total_questions = len(results)
        
        metrics = {
            'exact_match_accuracy': total_correct / total_questions,
            'weighted_score': total_weighted_score / total_possible_points,
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'total_questions': total_questions,
            'total_correct': total_correct,
            'difficulty_breakdown': self._calculate_difficulty_breakdown(results),
            'detailed_results': results
        }
        
        return metrics
    
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

def main():
    """
    Main function to run evaluation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate WTQ-style predictions with NORM and Mix self-consistency')
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON file')
    parser.add_argument('--ground-truths', required=True, help='Path to ground truths JSON file')
    parser.add_argument('--k-samples', type=int, default=5, help='Number of samples for Mix self-consistency')
    parser.add_argument('--output', help='Path to save evaluation results JSON')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = WTQEvaluator(k_samples=args.k_samples)
    results = evaluator.evaluate_answers(args.predictions, args.ground_truths)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Save results if output path provided
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()