import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_json_files(directory, model_name):
    """
    Load all JSON files from the specified directory into a DataFrame
    Add model name as a column for comparison
    """
    data = []
    for json_file in Path(directory).glob('*.json'):
        with open(json_file, 'r') as f:
            file_data = json.load(f)
            dataset_name = json_file.stem
            for item in file_data:
                for key, value in item.items():
                    if isinstance(value, str):
                        item[key] = value.lower()
                item['dataset'] = dataset_name
                item['model'] = model_name
            data.extend(file_data)
    return pd.DataFrame(data)

def create_difficulty_comparison_plot(df, models):
    """
    Create a bar plot comparing difficulties across datasets and models
    """
    plt.figure(figsize=(15, 8))
    
    # Calculate average scores by dataset, model, and difficulty
    difficulty_scores = df.pivot_table(
        index=['dataset', 'model'],
        columns='difficulty',
        values='score',
        aggfunc=['mean', 'count']
    )
    
    # Prepare data for plotting
    plot_data = []
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        for difficulty in ['easy', 'medium', 'hard']:
            for model in models:
                score = dataset_df[(dataset_df['model'] == model) & 
                                 (dataset_df['difficulty'] == difficulty)]['score'].mean()
                count = len(dataset_df[(dataset_df['model'] == model) & 
                                     (dataset_df['difficulty'] == difficulty)])
                plot_data.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Difficulty': difficulty,
                    'Score': score,
                    'Count': count
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped bar plot
    bar_width = 0.15
    x = np.arange(len(plot_df['Dataset'].unique()))
    
    for i, (model, difficulty) in enumerate([(m, d) for m in models for d in ['easy', 'medium', 'hard']]):
        mask = (plot_df['Model'] == model) & (plot_df['Difficulty'] == difficulty)
        scores = plot_df[mask]['Score']
        counts = plot_df[mask]['Count']
        
        bars = plt.bar(x + i*bar_width, scores, bar_width, 
                      label=f'{model} - {difficulty}')
        
        # Add count annotations
        for idx, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={int(count)}', ha='center', va='bottom', rotation=90)
    
    plt.xlabel('Dataset')
    plt.ylabel('Average Score')
    plt.title('Model Performance Comparison by Dataset and Difficulty')
    plt.xticks(x + bar_width * 2.5, plot_df['Dataset'].unique(), rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_report(df, models):
    """
    Create a single comprehensive report with all comparisons
    """
    with open('comprehensive_model_comparison.txt', 'w') as f:
        # Overall comparison
        f.write("=== Overall Comparison Between Models ===\n")
        
        # Overall statistics
        f.write("\nOverall Statistics:\n")
        for model in models:
            model_df = df[df['model'] == model]
            f.write(f"\n{model}:")
            f.write(f"\nTotal questions: {len(model_df)}")
            f.write(f"\nOverall accuracy: {model_df['score'].mean():.3f}")
            
            # Accuracy by difficulty
            f.write("\nAccuracy by difficulty:")
            diff_stats = model_df.groupby('difficulty')['score'].agg(['mean', 'count']).round(3)
            f.write("\n" + diff_stats.to_string())
            f.write("\n")
        
        # Type comparison with margins
        f.write("\n\nOverall Scores by Type and Difficulty - Model Comparison:\n")
        type_comparison = pd.pivot_table(
            df,
            index=['type'],
            columns=['model', 'difficulty'],
            values='score',
            aggfunc=['mean', 'count'],
            margins=True
        ).round(3)
        f.write(type_comparison.to_string())
        
        # Subtype comparison with margins
        f.write("\n\nOverall Scores by Type, Subtype and Difficulty - Model Comparison:\n")
        subtype_comparison = pd.pivot_table(
            df,
            index=['type', 'subtype'],
            columns=['model', 'difficulty'],
            values='score',
            aggfunc=['mean', 'count'],
            margins=True
        ).round(3)
        f.write(subtype_comparison.to_string())
        
        # Dataset-specific comparisons
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            
            f.write(f"\n\n=== Model Comparison for Dataset: {dataset} ===\n")
            
            # Type comparison
            f.write("\nScores by Type and Difficulty:\n")
            type_comparison = pd.pivot_table(
                dataset_df,
                index='type',
                columns=['model', 'difficulty'],
                values='score',
                aggfunc=['mean', 'count'],
                margins=True
            ).round(3)
            f.write(type_comparison.to_string())
            
            # Subtype comparison
            f.write("\n\nScores by Type, Subtype and Difficulty:\n")
            subtype_comparison = pd.pivot_table(
                dataset_df,
                index=['type', 'subtype'],
                columns=['model', 'difficulty'],
                values='score',
                aggfunc=['mean', 'count'],
                margins=True
            ).round(3)
            f.write(subtype_comparison.to_string())
            
            # Performance differences
            f.write("\n\nPerformance Differences:\n")
            for model1, model2 in zip(models[:-1], models[1:]):
                model1_scores = pd.pivot_table(
                    dataset_df[dataset_df['model'] == model1],
                    index='type',
                    columns='difficulty',
                    values='score',
                    aggfunc='mean'
                )
                model2_scores = pd.pivot_table(
                    dataset_df[dataset_df['model'] == model2],
                    index='type',
                    columns='difficulty',
                    values='score',
                    aggfunc='mean'
                )
                diff = (model2_scores - model1_scores).round(3)
                f.write(f"\nDifference ({model2} - {model1}):\n")
                f.write(diff.to_string())

def main():
    # Paths for both model results
    model1_dir = "./results_folder/claude-3.5-scores"
    model2_dir = "./results_folder/gpt-4o-scores"  
    
    try:
        # Load data from both models
        df1 = load_json_files(model1_dir, "claude-3.5")
        df2 = load_json_files(model2_dir, "gpt-4o")  
        
        # Combine dataframes
        combined_df = pd.concat([df1, df2])
        models = ["claude-3.5", "gpt-4o"]  
        
        # Create comprehensive report
        create_comprehensive_report(combined_df, models)
        
        # Create comparison plot
        create_difficulty_comparison_plot(combined_df, models)
        
        print("\nAnalysis complete! Check 'comprehensive_model_comparison.txt' for detailed comparisons")
        print("and 'model_comparison_plot.png' for the visualization.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()