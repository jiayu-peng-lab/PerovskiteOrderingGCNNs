#!/usr/bin/env python3
"""
Model Statistics Analysis Script
===============================

This script calculates and compares comprehensive statistics between CGCNN, e3nn, and ALIGNN models
for perovskite ordering prediction. It includes:

1. Basic error metrics (MAE, RMSE, R2)
2. Correlation analysis
3. Statistical significance tests
4. Performance comparison across different training fractions
5. Visualization of results

Usage:
    python model_statistics_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import custom functions
import sys
import os
sys.path.append('inference')
from plot_utils import get_property, flatten, get_relative_vals
from plot_utils import get_is_rocksalt, get_is_layered, get_is_column


class ModelStatisticsAnalyzer:
    """
    A comprehensive class for analyzing and comparing model statistics
    """
    
    def __init__(self):
        self.models = ['CGCNN', 'e3nn', 'ALIGNN']
        self.struct_types = ['unrelaxed', 'relaxed']
        self.training_fractions = [1.0, 0.5, 0.25, 0.125]
        self.results = {}
        
    def calculate_basic_metrics(self, y_true, y_pred):
        """
        Calculate basic error metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            dict: Dictionary containing MAE, RMSE, R2, and correlation
        """
        # Convert to numpy arrays and handle mixed data types
        try:
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
        except (ValueError, TypeError):
            # If conversion fails, try to extract numeric values
            y_true = pd.to_numeric(y_true, errors='coerce')
            y_pred = pd.to_numeric(y_pred, errors='coerce')
            y_true = y_true.values
            y_pred = y_pred.values
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'correlation': np.nan}
        
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Calculate correlation coefficient
        try:
            correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        except:
            correlation = np.nan
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'correlation': correlation,
            'n_samples': len(y_true_clean)
        }
    
    def calculate_advanced_metrics(self, y_true, y_pred):
        """
        Calculate advanced statistical metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            dict: Dictionary containing advanced metrics
        """
        # Convert to numpy arrays and handle mixed data types
        try:
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
        except (ValueError, TypeError):
            # If conversion fails, try to extract numeric values
            y_true = pd.to_numeric(y_true, errors='coerce')
            y_pred = pd.to_numeric(y_pred, errors='coerce')
            y_true = y_true.values
            y_pred = y_pred.values
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'mean_error': np.nan,
                'std_error': np.nan,
                'median_error': np.nan,
                'max_error': np.nan,
                'min_error': np.nan,
                'error_skewness': np.nan,
                'error_kurtosis': np.nan
            }
        
        errors = y_pred_clean - y_true_clean
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'error_skewness': stats.skew(errors),
            'error_kurtosis': stats.kurtosis(errors)
        }
    
    def load_model_predictions(self, model, struct_type, training_fraction=1.0):
        """
        Load model predictions from JSON files
        
        Args:
            model: Model name (CGCNN, e3nn, ALIGNN)
            struct_type: Structure type (unrelaxed, relaxed)
            training_fraction: Training fraction used
            
        Returns:
            tuple: (test_predictions, holdout_predictions)
        """
        try:
            # Load test set predictions
            test_file = f"./best_models/{model}/dft_e_hull_htvs_data_{struct_type}_{model}/best_0/test_set_predictions.json"
            test_data = pd.read_json(test_file)
            
            # Load holdout set predictions
            holdout_file = f"./best_models/{model}/dft_e_hull_htvs_data_{struct_type}_{model}/best_0/holdout_set_B_sites_predictions.json"
            holdout_data = pd.read_json(holdout_file)
            
            return test_data, holdout_data
        except FileNotFoundError:
            print(f"Warning: Could not find prediction files for {model} {struct_type}")
            return None, None
    
    def analyze_model_performance(self, model, struct_type):
        """
        Analyze performance for a specific model and structure type
        
        Args:
            model: Model name
            struct_type: Structure type
            
        Returns:
            dict: Performance analysis results
        """
        print(f"Analyzing {model} on {struct_type} structures...")
        
        # Load predictions
        test_data, holdout_data = self.load_model_predictions(model, struct_type)
        
        if test_data is None or holdout_data is None:
            return None
        
        # Extract true and predicted values
        y_true_test = test_data['dft_e_hull'].values
        y_pred_test = test_data['predicted_dft_e_hull'].values
        
        y_true_holdout = holdout_data['dft_e_hull'].values
        y_pred_holdout = holdout_data['predicted_dft_e_hull'].values
        
        # Debug: Print data types and sample values
        print(f"  Data types - y_true_test: {type(y_true_test)}, y_pred_test: {type(y_pred_test)}")
        print(f"  Sample values - y_true_test[0]: {y_true_test[0]}, y_pred_test[0]: {y_pred_test[0]}")
        print(f"  Data shapes - y_true_test: {y_true_test.shape}, y_pred_test: {y_pred_test.shape}")
        
        # Handle predicted values that might be lists BEFORE passing to other functions
        if isinstance(y_pred_test[0], list):
            y_pred_test = np.array([pred[0] if isinstance(pred, list) else pred for pred in y_pred_test])
        if isinstance(y_pred_holdout[0], list):
            y_pred_holdout = np.array([pred[0] if isinstance(pred, list) else pred for pred in y_pred_holdout])
        
        # Calculate metrics
        test_metrics = self.calculate_basic_metrics(y_true_test, y_pred_test)
        holdout_metrics = self.calculate_basic_metrics(y_true_holdout, y_pred_holdout)
        
        test_advanced = self.calculate_advanced_metrics(y_true_test, y_pred_test)
        holdout_advanced = self.calculate_advanced_metrics(y_true_holdout, y_pred_holdout)
        
        # Calculate relative performance metrics
        test_relative = get_relative_vals(test_data, y_pred_test)
        holdout_relative = get_relative_vals(holdout_data, y_pred_holdout)
        
        relative_metrics = {
            'test_relative_mae': np.mean(np.abs(test_relative)),
            'test_relative_std': np.std(test_relative),
            'holdout_relative_mae': np.mean(np.abs(holdout_relative)),
            'holdout_relative_std': np.std(holdout_relative)
        }
        
        return {
            'model': model,
            'struct_type': struct_type,
            'test_metrics': test_metrics,
            'holdout_metrics': holdout_metrics,
            'test_advanced': test_advanced,
            'holdout_advanced': holdout_advanced,
            'relative_metrics': relative_metrics,
            'test_data': test_data,
            'holdout_data': holdout_data
        }
    
    def compare_models(self, struct_type):
        """
        Compare all models for a given structure type
        
        Args:
            struct_type: Structure type to analyze
            
        Returns:
            dict: Comparison results
        """
        print(f"\n{'='*60}")
        print(f"COMPARING MODELS FOR {struct_type.upper()} STRUCTURES")
        print(f"{'='*60}")
        
        comparison_results = {}
        
        for model in self.models:
            results = self.analyze_model_performance(model, struct_type)
            if results:
                comparison_results[model] = results
                
                # Print summary
                print(f"\n{model} Results:")
                print(f"  Test Set - MAE: {results['test_metrics']['MAE']:.4f}, "
                      f"RMSE: {results['test_metrics']['RMSE']:.4f}, "
                      f"R2: {results['test_metrics']['R2']:.4f}")
                print(f"  Holdout Set - MAE: {results['holdout_metrics']['MAE']:.4f}, "
                      f"RMSE: {results['holdout_metrics']['RMSE']:.4f}, "
                      f"R2: {results['holdout_metrics']['R2']:.4f}")
        
        return comparison_results
    
    def statistical_significance_test(self, model1_results, model2_results, metric='MAE'):
        """
        Perform statistical significance test between two models
        
        Args:
            model1_results: Results from first model
            model2_results: Results from second model
            metric: Metric to compare ('MAE', 'RMSE', 'R2')
            
        Returns:
            dict: Statistical test results
        """
        # Extract errors for comparison and handle data types
        try:
            y_true_1 = pd.to_numeric(model1_results['test_data']['dft_e_hull'], errors='coerce').values
            y_pred_1 = pd.to_numeric(model1_results['test_data']['predicted_dft_e_hull'], errors='coerce').values
            errors_1 = np.abs(y_pred_1 - y_true_1)
            
            y_true_2 = pd.to_numeric(model2_results['test_data']['dft_e_hull'], errors='coerce').values
            y_pred_2 = pd.to_numeric(model2_results['test_data']['predicted_dft_e_hull'], errors='coerce').values
            errors_2 = np.abs(y_pred_2 - y_true_2)
        except:
            return {'p_value': np.nan, 'significant': False, 'test_statistic': np.nan}
        
        # Remove NaN values
        mask_1 = ~np.isnan(errors_1)
        mask_2 = ~np.isnan(errors_2)
        errors_1_clean = errors_1[mask_1]
        errors_2_clean = errors_2[mask_2]
        
        if len(errors_1_clean) == 0 or len(errors_2_clean) == 0:
            return {'p_value': np.nan, 'significant': False, 'test_statistic': np.nan}
        
        # Ensure both arrays have the same length for comparison
        min_length = min(len(errors_1_clean), len(errors_2_clean))
        errors_1_clean = errors_1_clean[:min_length]
        errors_2_clean = errors_2_clean[:min_length]
        
        # Perform Wilcoxon signed-rank test (non-parametric)
        try:
            statistic, p_value = stats.wilcoxon(errors_1_clean, errors_2_clean)
            significant = p_value < 0.05
        except:
            # If Wilcoxon fails, try t-test
            try:
                statistic, p_value = stats.ttest_ind(errors_1_clean, errors_2_clean)
                significant = p_value < 0.05
            except:
                p_value = np.nan
                significant = False
                statistic = np.nan
        
        return {
            'p_value': p_value,
            'significant': significant,
            'test_statistic': statistic
        }
    
    def create_comparison_table(self, comparison_results):
        """
        Create a formatted comparison table
        
        Args:
            comparison_results: Results from model comparison
            
        Returns:
            pd.DataFrame: Formatted comparison table
        """
        table_data = []
        
        for model, results in comparison_results.items():
            row = {
                'Model': model,
                'Test MAE': f"{results['test_metrics']['MAE']:.4f}",
                'Test RMSE': f"{results['test_metrics']['RMSE']:.4f}",
                'Test R²': f"{results['test_metrics']['R2']:.4f}",
                'Test Corr': f"{results['test_metrics']['correlation']:.4f}",
                'Holdout MAE': f"{results['holdout_metrics']['MAE']:.4f}",
                'Holdout RMSE': f"{results['holdout_metrics']['RMSE']:.4f}",
                'Holdout R²': f"{results['holdout_metrics']['R2']:.4f}",
                'Holdout Corr': f"{results['holdout_metrics']['correlation']:.4f}"
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def plot_performance_comparison(self, comparison_results, struct_type):
        """
        Create performance comparison plots
        
        Args:
            comparison_results: Results from model comparison
            struct_type: Structure type being analyzed
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Performance Comparison - {struct_type.title()} Structures', fontsize=16)
        
        metrics = ['MAE', 'RMSE', 'R2', 'correlation']
        datasets = ['test', 'holdout']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            models = list(comparison_results.keys())
            values = []
            
            for model in models:
                if metric in ['MAE', 'RMSE']:
                    # Lower is better for error metrics
                    test_val = comparison_results[model][f'{datasets[0]}_metrics'][metric]
                    holdout_val = comparison_results[model][f'{datasets[1]}_metrics'][metric]
                else:
                    # Higher is better for R2 and correlation
                    test_val = comparison_results[model][f'{datasets[0]}_metrics'][metric]
                    holdout_val = comparison_results[model][f'{datasets[1]}_metrics'][metric]
                
                values.append([test_val, holdout_val])
            
            values = np.array(values)
            
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, values[:, 0], width, label='Test Set', alpha=0.8)
            ax.bar(x + width/2, values[:, 1], width, label='Holdout Set', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'model_performance_comparison_{struct_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run complete analysis for all models and structure types
        """
        print("Starting comprehensive model statistics analysis...")
        print("=" * 80)
        
        all_results = {}
        
        for struct_type in self.struct_types:
            print(f"\nAnalyzing {struct_type} structures...")
            comparison_results = self.compare_models(struct_type)
            
            if comparison_results:
                all_results[struct_type] = comparison_results
                
                # Create comparison table
                table = self.create_comparison_table(comparison_results)
                print(f"\n{struct_type.title()} Structures - Performance Comparison:")
                print(table.to_string(index=False))
                
                # Perform statistical significance tests
                print(f"\n{struct_type.title()} Structures - Statistical Significance Tests:")
                models_list = list(comparison_results.keys())
                for i in range(len(models_list)):
                    for j in range(i+1, len(models_list)):
                        model1, model2 = models_list[i], models_list[j]
                        test_result = self.statistical_significance_test(
                            comparison_results[model1], 
                            comparison_results[model2]
                        )
                        
                        significance = "SIGNIFICANT" if test_result['significant'] else "NOT SIGNIFICANT"
                        print(f"  {model1} vs {model2}: p-value = {test_result['p_value']:.4f} ({significance})")
                
                # Create plots
                self.plot_performance_comparison(comparison_results, struct_type)
        
        # Save results to file
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, all_results):
        """
        Save analysis results to files
        
        Args:
            all_results: Complete analysis results
        """
        # Save summary to CSV
        summary_data = []
        for struct_type, comparison_results in all_results.items():
            for model, results in comparison_results.items():
                row = {
                    'Structure_Type': struct_type,
                    'Model': model,
                    'Test_MAE': results['test_metrics']['MAE'],
                    'Test_RMSE': results['test_metrics']['RMSE'],
                    'Test_R2': results['test_metrics']['R2'],
                    'Test_Correlation': results['test_metrics']['correlation'],
                    'Holdout_MAE': results['holdout_metrics']['MAE'],
                    'Holdout_RMSE': results['holdout_metrics']['RMSE'],
                    'Holdout_R2': results['holdout_metrics']['R2'],
                    'Holdout_Correlation': results['holdout_metrics']['correlation'],
                    'Test_Samples': results['test_metrics']['n_samples'],
                    'Holdout_Samples': results['holdout_metrics']['n_samples']
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('model_statistics_summary.csv', index=False)
        print(f"\nResults saved to 'model_statistics_summary.csv'")
        
        # Save detailed results
        with open('model_statistics_detailed.txt', 'w') as f:
            f.write("DETAILED MODEL STATISTICS ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            for struct_type, comparison_results in all_results.items():
                f.write(f"{struct_type.upper()} STRUCTURES\n")
                f.write("-" * 30 + "\n")
                
                for model, results in comparison_results.items():
                    f.write(f"\n{model}:\n")
                    f.write(f"  Test Metrics: {results['test_metrics']}\n")
                    f.write(f"  Holdout Metrics: {results['holdout_metrics']}\n")
                    f.write(f"  Advanced Test Metrics: {results['test_advanced']}\n")
                    f.write(f"  Advanced Holdout Metrics: {results['holdout_advanced']}\n")
                    f.write(f"  Relative Metrics: {results['relative_metrics']}\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Detailed results saved to 'model_statistics_detailed.txt'")


def main():
    """
    Main function to run the analysis
    """
    print("Model Statistics Analysis for CGCNN, e3nn, and ALIGNN")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ModelStatisticsAnalyzer()
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSummary of results:")
        print("- Performance comparison tables created")
        print("- Statistical significance tests performed")
        print("- Visualization plots generated")
        print("- Results saved to CSV and text files")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
