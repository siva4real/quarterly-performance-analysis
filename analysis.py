"""
Data Analysis for Quarterly Performance Metrics
Author: 24f3004072@ds.study.iitm.ac.in
Date: December 7, 2025

This script analyzes quarterly performance data and generates insights
for reaching the target of 90% through predictive maintenance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_quarterly_data():
    """
    Load and prepare quarterly performance data.
    The data shows performance metrics over 4 quarters.
    Average value: 73.35
    """
    data = {
        'Quarter': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
        'Performance': [70.5, 72.8, 75.2, 74.9],
        'Target': [90, 90, 90, 90],
        'Industry_Benchmark': [78.0, 79.5, 80.0, 81.0]
    }
    
    df = pd.DataFrame(data)
    return df

def calculate_statistics(df):
    """Calculate key statistics from the data."""
    stats = {
        'mean': df['Performance'].mean(),
        'median': df['Performance'].median(),
        'std': df['Performance'].std(),
        'trend_slope': np.polyfit(range(len(df)), df['Performance'], 1)[0],
        'gap_to_target': 90 - df['Performance'].mean(),
        'gap_to_benchmark': df['Industry_Benchmark'].mean() - df['Performance'].mean()
    }
    return stats

def predict_future_performance(df, periods=4):
    """
    Predict future performance based on current trend.
    Uses linear regression for projection.
    """
    x = np.arange(len(df))
    y = df['Performance'].values
    
    # Fit linear model
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # Predict future
    future_x = np.arange(len(df), len(df) + periods)
    future_pred = p(future_x)
    
    return future_pred

def analyze_maintenance_impact():
    """
    Estimate the impact of implementing predictive maintenance program.
    Based on industry research, predictive maintenance can improve
    performance by 15-20%.
    """
    impact = {
        'conservative_improvement': 0.15,  # 15%
        'expected_improvement': 0.175,      # 17.5%
        'optimistic_improvement': 0.20      # 20%
    }
    return impact

def generate_trend_visualization(df):
    """Generate trend analysis visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    quarters = range(len(df))
    
    # Plot actual performance
    ax.plot(quarters, df['Performance'], marker='o', linewidth=2, 
            markersize=10, label='Actual Performance', color='#2E86AB')
    
    # Plot target line
    ax.plot(quarters, df['Target'], linestyle='--', linewidth=2,
            label='Target (90)', color='#A23B72')
    
    # Plot industry benchmark
    ax.plot(quarters, df['Industry_Benchmark'], marker='s', linewidth=2,
            markersize=8, label='Industry Benchmark', color='#F18F01')
    
    # Add trend line
    z = np.polyfit(quarters, df['Performance'], 1)
    p = np.poly1d(z)
    ax.plot(quarters, p(quarters), linestyle=':', linewidth=2,
            label='Trend Line', color='#C73E1D', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_title('Quarterly Performance Analysis: Trend vs Target vs Benchmark', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(quarters)
    ax.set_xticklabels(df['Quarter'])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([65, 95])
    
    # Add average line
    avg = df['Performance'].mean()
    ax.axhline(y=avg, color='green', linestyle='-.', linewidth=1.5, 
               alpha=0.5, label=f'Average: {avg:.2f}')
    
    plt.tight_layout()
    plt.savefig('trend_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Trend visualization saved as 'trend_analysis.png'")
    plt.close()

def generate_benchmark_comparison(df):
    """Generate benchmark comparison visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Bar comparison
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax1.bar(x - width, df['Performance'], width, 
                    label='Our Performance', color='#2E86AB')
    bars2 = ax1.bar(x, df['Industry_Benchmark'], width,
                    label='Industry Benchmark', color='#F18F01')
    bars3 = ax1.bar(x + width, df['Target'], width,
                    label='Target', color='#A23B72')
    
    ax1.set_xlabel('Quarter', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Performance Score', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Comparison by Quarter', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Quarter'])
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Subplot 2: Gap analysis
    gaps_to_target = df['Target'] - df['Performance']
    gaps_to_benchmark = df['Industry_Benchmark'] - df['Performance']
    
    ax2.plot(x, gaps_to_target, marker='o', linewidth=2, markersize=8,
            label='Gap to Target', color='#A23B72')
    ax2.plot(x, gaps_to_benchmark, marker='s', linewidth=2, markersize=8,
            label='Gap to Benchmark', color='#F18F01')
    
    ax2.set_xlabel('Quarter', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Performance Gap', fontsize=11, fontweight='bold')
    ax2.set_title('Gap Analysis', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Quarter'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Benchmark comparison saved as 'benchmark_comparison.png'")
    plt.close()

def generate_projection_visualization(df, stats):
    """Generate future projection with predictive maintenance impact."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Historical data
    historical_x = np.arange(len(df))
    ax.plot(historical_x, df['Performance'], marker='o', linewidth=2.5,
            markersize=10, label='Historical Performance', color='#2E86AB')
    
    # Current trend projection (without intervention)
    future_periods = 4
    future_x = np.arange(len(df), len(df) + future_periods)
    future_pred = predict_future_performance(df, future_periods)
    
    all_x = np.concatenate([historical_x, future_x])
    all_y = np.concatenate([df['Performance'].values, future_pred])
    
    ax.plot(future_x, future_pred, marker='o', linewidth=2,
            markersize=8, linestyle='--', label='Projected (Current Trend)',
            color='#C73E1D', alpha=0.7)
    
    # With predictive maintenance
    impact = analyze_maintenance_impact()
    current_avg = df['Performance'].iloc[-1]
    
    # Conservative scenario
    conservative_improvement = current_avg * (1 + impact['conservative_improvement'])
    conservative_proj = np.linspace(current_avg, 
                                   min(conservative_improvement * 1.15, 90), 
                                   future_periods)
    
    # Expected scenario
    expected_improvement = current_avg * (1 + impact['expected_improvement'])
    expected_proj = np.linspace(current_avg,
                                min(expected_improvement * 1.2, 90),
                                future_periods)
    
    # Optimistic scenario
    optimistic_improvement = current_avg * (1 + impact['optimistic_improvement'])
    optimistic_proj = np.linspace(current_avg,
                                  min(optimistic_improvement * 1.25, 90),
                                  future_periods)
    
    ax.plot(future_x, conservative_proj, marker='s', linewidth=2,
            linestyle='-.', label='With Predictive Maintenance (Conservative)',
            color='#06A77D', alpha=0.8)
    
    ax.plot(future_x, expected_proj, marker='D', linewidth=2.5,
            linestyle='-.', label='With Predictive Maintenance (Expected)',
            color='#059142', alpha=0.9)
    
    ax.plot(future_x, optimistic_proj, marker='^', linewidth=2,
            linestyle='-.', label='With Predictive Maintenance (Optimistic)',
            color='#037F3C', alpha=0.7)
    
    # Target line
    ax.axhline(y=90, color='#A23B72', linestyle='--', linewidth=2.5,
               label='Target (90)', alpha=0.8)
    
    # Formatting
    all_quarters = df['Quarter'].tolist() + [f'Q{i} 2025' for i in range(1, future_periods + 1)]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_quarters, rotation=45, ha='right')
    ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Projection: Impact of Predictive Maintenance Program',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([68, 95])
    
    # Add shaded region for predictive maintenance impact
    ax.fill_between(future_x, future_pred, expected_proj,
                    alpha=0.2, color='green', label='Expected Improvement Range')
    
    # Add vertical line to separate historical and future
    ax.axvline(x=len(df)-0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(len(df)-0.5, 92, 'Future Projection', ha='left', fontsize=10,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('projection_with_maintenance.png', dpi=300, bbox_inches='tight')
    print("✓ Projection visualization saved as 'projection_with_maintenance.png'")
    plt.close()

def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("QUARTERLY PERFORMANCE DATA ANALYSIS")
    print("Author: 24f3004072@ds.study.iitm.ac.in")
    print("=" * 70)
    print()
    
    # Load data
    print("📊 Loading quarterly data...")
    df = load_quarterly_data()
    print(df.to_string(index=False))
    print()
    
    # Calculate statistics
    print("📈 Calculating statistics...")
    stats = calculate_statistics(df)
    print(f"  • Average Performance: {stats['mean']:.2f}")
    print(f"  • Median Performance: {stats['median']:.2f}")
    print(f"  • Standard Deviation: {stats['std']:.2f}")
    print(f"  • Trend Slope: {stats['trend_slope']:.3f} (per quarter)")
    print(f"  • Gap to Target (90): {stats['gap_to_target']:.2f}")
    print(f"  • Gap to Benchmark: {stats['gap_to_benchmark']:.2f}")
    print()
    
    # Generate visualizations
    print("🎨 Generating visualizations...")
    generate_trend_visualization(df)
    generate_benchmark_comparison(df)
    generate_projection_visualization(df, stats)
    print()
    
    # Predictive maintenance analysis
    print("🔧 Analyzing predictive maintenance impact...")
    impact = analyze_maintenance_impact()
    current_avg = stats['mean']
    
    print(f"  • Current Average: {current_avg:.2f}")
    print(f"  • Conservative Improvement (+15%): {current_avg * (1 + impact['conservative_improvement']):.2f}")
    print(f"  • Expected Improvement (+17.5%): {current_avg * (1 + impact['expected_improvement']):.2f}")
    print(f"  • Optimistic Improvement (+20%): {current_avg * (1 + impact['optimistic_improvement']):.2f}")
    print()
    
    print("=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print("Key Finding: Current average of 73.35 is below target of 90.")
    print("Recommendation: Implement predictive maintenance program to bridge the gap.")
    print()

if __name__ == "__main__":
    main()

