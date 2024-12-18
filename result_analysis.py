import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_json_files(model_paths):
    """
    Load JSON files from multiple model directories
    model_paths: dict of {model_name: path_to_results}
    """
    all_data = []
    
    for model_name, directory in model_paths.items():
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
        all_data.extend(data)
    
    return pd.DataFrame(all_data)

def create_difficulty_comparison_plot(df):
    """
    Create a bar plot comparing difficulties across datasets and models
    """
    plt.figure(figsize=(15, 8))
    
    models = df['model'].unique()
    difficulties = ['easy', 'medium', 'hard']
    
    # Prepare data for plotting
    plot_data = []
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        for difficulty in difficulties:
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
    bar_width = 0.8 / (len(models) * len(difficulties))  # Adjust bar width based on number of models
    x = np.arange(len(plot_df['Dataset'].unique()))
    
    for i, (model, difficulty) in enumerate([(m, d) for m in models for d in difficulties]):
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
    plt.xticks(x + (bar_width * len(models) * len(difficulties))/2, 
               plot_df['Dataset'].unique(), rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def clean_dataset_name(dataset):
    """
    Extract the common part of dataset names by removing model-specific parts
    """
    parts = dataset.split('_')
    return '_'.join(parts[:-1])  # Remove the last part which is model-specific

def create_dataset_performance_plot(df):
    """
    Create separate plots for each difficulty level comparing model performance on datasets
    """
    # Clean dataset names
    df['clean_dataset'] = df['dataset'].apply(clean_dataset_name)
    
    difficulties = ['easy', 'medium', 'hard']
    models = sorted(df['model'].unique())
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    for idx, difficulty in enumerate(difficulties):
        # Filter data for this difficulty
        diff_df = df[df['difficulty'] == difficulty]
        
        # Calculate scores for each model and dataset
        dataset_scores = diff_df.pivot_table(
            index='clean_dataset',
            columns='model',
            values='score',
            aggfunc=['mean', 'count']
        ).round(3)
        
        # Plot bars
        ax = axes[idx]
        bar_width = 0.8 / len(models)
        x = np.arange(len(dataset_scores.index))
        
        for i, model in enumerate(models):
            scores = dataset_scores['mean'][model]
            counts = dataset_scores['count'][model]
            
            bars = ax.bar(x + i*bar_width, scores, bar_width, label=model)
            
            # Add count annotations
            for j, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'n={int(count)}', ha='center', va='bottom')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Model Performance Comparison - {difficulty.upper()} Questions')
        ax.set_xticks(x + (bar_width * len(models))/2)
        ax.set_xticklabels(dataset_scores.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limits from 0 to 1
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('dataset_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_report(df):
    """
    Create a single comprehensive report with all comparisons
    """
    models = sorted(df['model'].unique())
    
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
        for dataset in sorted(df['dataset'].unique()):
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
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    model1, model2 = models[i], models[j]
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
        
        # Add direct dataset comparison
        f.write("\n\n=== Direct Dataset Performance Comparison ===\n")
        
        # Overall dataset comparison
        dataset_comparison = pd.pivot_table(
            df,
            index='dataset',
            columns='model',
            values='score',
            aggfunc=['mean', 'count']
        ).round(3)
        f.write("\nOverall Performance by Dataset:\n")
        f.write(dataset_comparison.to_string())
        
        # Calculate and show all pairwise performance differences
        f.write("\n\nPairwise Performance Differences by Dataset:\n")
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model1, model2 = models[i], models[j]
                diff_scores = []
                for dataset in sorted(df['dataset'].unique()):
                    model1_score = df[(df['model'] == model1) & 
                                    (df['dataset'] == dataset)]['score'].mean()
                    model2_score = df[(df['model'] == model2) & 
                                    (df['dataset'] == dataset)]['score'].mean()
                    diff_scores.append({
                        'Dataset': dataset,
                        'Difference': round(model2_score - model1_score, 3)
                    })
                diff_df = pd.DataFrame(diff_scores)
                f.write(f"\n{model2} - {model1}:\n")
                f.write(diff_df.to_string())

def main():
    # Define model paths as a dictionary
    model_paths = {
        "claude-3.5": "./results_folder/claude-3.5-scores",
        "gpt-4o": "./results_folder/gpt-4o-scores",
        "gpt-3.5-turbo": "./results_folder/gpt-3.5-turbo-scores",
        "gpt-3.5-turbo": "./results_folder/gpt-3.5-turbo-run2-scores",

        # Add more models as needed:
        # "model-3": "./path/to/model3/scores",
        # "model-4": "./path/to/model4/scores",
    }
    
    try:
        # Load data from all models
        combined_df = load_json_files(model_paths)
        
        # Create comprehensive report
        create_comprehensive_report(combined_df)
        
        # Create comparison plots
        create_difficulty_comparison_plot(combined_df)
        create_dataset_performance_plot(combined_df)
        
        print("\nAnalysis complete! Check:")
        print("- 'comprehensive_model_comparison.txt' for detailed comparisons")
        print("- 'model_comparison_plot.png' for difficulty comparison")
        print("- 'dataset_performance_comparison.png' for dataset comparison")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()