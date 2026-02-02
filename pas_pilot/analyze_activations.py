"""
PAS Activations Analysis Script

이 스크립트는 PAS (Persona Activation Steering) 방법으로 생성된 activations를 분석합니다.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import os
from sklearn.metrics.pairwise import cosine_similarity

# Matplotlib 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_activations(filepath):
    """Activations 파일 로드"""
    print("=" * 80)
    print("1. LOADING ACTIVATIONS FILE")
    print("=" * 80)
    
    with open(filepath, 'rb') as f:
        activations_list = pickle.load(f)
    
    print(f"\n✓ File loaded: {filepath}")
    print(f"✓ Total samples loaded: {len(activations_list)}")
    print(f"✓ Data type: {type(activations_list)}")
    
    print(f'raw data: {activations_list}')
    return activations_list


def check_data_structure(activations_list):
    """데이터 구조 확인"""
    print("\n" + "=" * 80)
    print("2. DATA STRUCTURE ANALYSIS")
    print("=" * 80)
    
    if len(activations_list) == 0:
        print("\n✗ WARNING: No samples in the activations list!")
        print("✗ This indicates that all samples were filtered out during processing.")
        return False
    
    print(f"\n✓ First sample structure:")
    print(f"  Keys: {list(activations_list[0].keys())}")
    print(f"\n✓ Case: {activations_list[0]['case']}")
    print(f"\n✓ System prompt (first 200 chars):")
    print(f"  {activations_list[0]['system_prompt'][:200]}...")
    
    return True


def analyze_activations_detail(activations_list):
    """Activations 상세 분석"""
    print("\n" + "=" * 80)
    print("3. ACTIVATIONS DETAILED ANALYSIS")
    print("=" * 80)
    
    if len(activations_list) == 0:
        print("\n✗ No data to analyze.")
        return None
    
    first_activations = activations_list[0]['activations']
    
    print(f"\n✓ Number of layers with interventions: {len(first_activations)}")
    print(f"\n✓ Layer names:")
    for i, layer_name in enumerate(first_activations.keys()):
        print(f"  {i+1}. {layer_name}")
    
    print(f"\n{'-'*80}")
    print("DETAILED LAYER INFORMATION (first 3 layers):")
    print(f"{'-'*80}")
    
    for layer_name, interventions in list(first_activations.items())[:3]:
        print(f"\n{layer_name}:")
        print(f"  Number of heads: {len(interventions)}")
        
        for i, (head_no, direction, std) in enumerate(interventions[:2]):  # 처음 2개 헤드만
            print(f"\n  Head {head_no}:")
            print(f"    Direction shape: {direction.shape if hasattr(direction, 'shape') else len(direction)}")
            print(f"    Direction norm: {np.linalg.norm(direction):.4f}")
            print(f"    Std value: {std:.4f}")
            print(f"    Direction sample (first 5): {direction[:5]}")
    
    return first_activations


def compute_layer_statistics(activations_list):
    """레이어별 통계 계산"""
    print("\n" + "=" * 80)
    print("4. LAYER STATISTICS")
    print("=" * 80)
    
    if len(activations_list) == 0:
        print("\n✗ No data to analyze.")
        return None, None
    
    layer_stats = {}
    
    for sample_idx, sample in enumerate(activations_list):
        activations = sample['activations']
        
        for layer_name, interventions in activations.items():
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    'num_heads': [],
                    'std_values': [],
                    'direction_norms': []
                }
            
            layer_stats[layer_name]['num_heads'].append(len(interventions))
            
            for head_no, direction, std in interventions:
                layer_stats[layer_name]['std_values'].append(std)
                layer_stats[layer_name]['direction_norms'].append(np.linalg.norm(direction))
    
    # DataFrame으로 변환
    stats_summary = []
    for layer_name, stats in layer_stats.items():
        stats_summary.append({
            'Layer': layer_name,
            'Avg Heads': np.mean(stats['num_heads']),
            'Avg Std': np.mean(stats['std_values']),
            'Std of Std': np.std(stats['std_values']),
            'Avg Direction Norm': np.mean(stats['direction_norms']),
            'Min Std': np.min(stats['std_values']),
            'Max Std': np.max(stats['std_values'])
        })
    
    df_stats = pd.DataFrame(stats_summary)
    print("\n" + df_stats.to_string(index=False))
    
    return layer_stats, df_stats


def plot_std_distribution(layer_stats, output_dir='./activations'):
    """Std 값 분포 시각화"""
    print("\n" + "=" * 80)
    print("5. STD VALUES VISUALIZATION")
    print("=" * 80)
    
    if not layer_stats:
        print("\n✗ No data to visualize.")
        return
    
    # 모든 std 값 수집
    all_stds = []
    for stats in layer_stats.values():
        all_stds.extend(stats['std_values'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(all_stds, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Std Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Std Values Across All Layers', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by layer
    layer_names = list(layer_stats.keys())
    std_by_layer = [layer_stats[ln]['std_values'] for ln in layer_names]
    
    bp = axes[1].boxplot(std_by_layer, labels=[f"L{i}" for i in range(len(layer_names))], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Std Value', fontsize=12)
    axes[1].set_title('Std Values by Layer', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'std_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")
    plt.close()
    
    print(f"\n✓ Std Statistics:")
    print(f"  Total values: {len(all_stds)}")
    print(f"  Mean: {np.mean(all_stds):.4f}")
    print(f"  Std: {np.std(all_stds):.4f}")
    print(f"  Min: {np.min(all_stds):.4f}")
    print(f"  Max: {np.max(all_stds):.4f}")


def plot_direction_norm_distribution(layer_stats, output_dir='./activations'):
    """Direction Norm 분포 시각화"""
    print("\n" + "=" * 80)
    print("6. DIRECTION NORM VISUALIZATION")
    print("=" * 80)
    
    if not layer_stats:
        print("\n✗ No data to visualize.")
        return
    
    # 모든 direction norm 수집
    all_norms = []
    for stats in layer_stats.values():
        all_norms.extend(stats['direction_norms'])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.hist(all_norms, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Direction Norm', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Direction Norms Across All Layers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 평균선 추가
    mean_norm = np.mean(all_norms)
    ax.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.4f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(output_dir, 'direction_norm_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")
    plt.close()
    
    print(f"\n✓ Direction Norm Statistics:")
    print(f"  Mean: {np.mean(all_norms):.4f}")
    print(f"  Std: {np.std(all_norms):.4f}")
    print(f"  Min: {np.min(all_norms):.4f}")
    print(f"  Max: {np.max(all_norms):.4f}")


def plot_heads_per_layer(layer_stats, output_dir='./activations'):
    """레이어별 헤드 수 시각화"""
    print("\n" + "=" * 80)
    print("7. HEADS PER LAYER VISUALIZATION")
    print("=" * 80)
    
    if not layer_stats:
        print("\n✗ No data to visualize.")
        return
    
    # 레이어별 평균 헤드 수
    layer_names = []
    avg_heads = []
    
    for layer_name, stats in layer_stats.items():
        layer_num = int(layer_name.split('.')[2])  # Extract layer number
        layer_names.append(layer_num)
        avg_heads.append(np.mean(stats['num_heads']))
    
    # 정렬
    sorted_indices = np.argsort(layer_names)
    layer_names = [layer_names[i] for i in sorted_indices]
    avg_heads = [avg_heads[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(range(len(layer_names)), avg_heads, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Layer Number', fontsize=12)
    ax.set_ylabel('Average Number of Heads', fontsize=12)
    ax.set_title('Average Number of Intervention Heads per Layer', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 평균선 추가
    mean_heads = np.mean(avg_heads)
    ax.axhline(mean_heads, color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {mean_heads:.2f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(output_dir, 'heads_per_layer.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")
    plt.close()


def compute_sample_summary(activations_list):
    """샘플별 요약 정보"""
    print("\n" + "=" * 80)
    print("8. SAMPLE SUMMARY")
    print("=" * 80)
    
    if len(activations_list) == 0:
        print("\n✗ No samples to summarize.")
        return None
    
    sample_summary = []
    
    for idx, sample in enumerate(activations_list):
        activations = sample['activations']
        
        total_interventions = sum(len(interventions) for interventions in activations.values())
        
        all_stds_sample = []
        for interventions in activations.values():
            for head_no, direction, std in interventions:
                all_stds_sample.append(std)
        
        sample_summary.append({
            'Sample': idx + 1,
            'Case': sample['case'],
            'Num Layers': len(activations),
            'Total Interventions': total_interventions,
            'Avg Std': np.mean(all_stds_sample) if all_stds_sample else 0,
            'System Prompt Length': len(sample['system_prompt'])
        })
    
    df_samples = pd.DataFrame(sample_summary)
    print("\n" + df_samples.to_string(index=False))
    
    return df_samples


def analyze_direction_similarity(activations_list, output_dir='./activations'):
    """Direction 벡터 간 유사도 분석"""
    print("\n" + "=" * 80)
    print("9. DIRECTION VECTOR SIMILARITY ANALYSIS")
    print("=" * 80)
    
    if len(activations_list) == 0:
        print("\n✗ No data to analyze.")
        return
    
    # 첫 번째 샘플의 첫 번째 레이어의 모든 방향 벡터 수집
    first_activations = activations_list[0]['activations']
    first_layer = list(first_activations.keys())[0]
    directions = []
    
    for head_no, direction, std in first_activations[first_layer]:
        directions.append(direction)
    
    directions = np.array(directions)
    
    print(f"\n✓ Analyzing direction vectors from {first_layer}")
    print(f"  Number of directions: {len(directions)}")
    print(f"  Direction dimension: {directions.shape[1]}")
    
    # Compute cosine similarities
    similarities = cosine_similarity(directions)
    
    # Visualize similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarities, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Head Index', fontsize=12)
    ax.set_title(f'Cosine Similarity between Direction Vectors\n{first_layer}', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)
    
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(output_dir, 'direction_similarity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")
    plt.close()
    
    # Statistics (대각선 제외)
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    off_diagonal = similarities[mask]
    
    print(f"\n✓ Similarity Statistics (off-diagonal):")
    print(f"  Mean: {np.mean(off_diagonal):.4f}")
    print(f"  Std: {np.std(off_diagonal):.4f}")
    print(f"  Min: {np.min(off_diagonal):.4f}")
    print(f"  Max: {np.max(off_diagonal):.4f}")


def export_to_csv(df_stats, df_samples, output_dir='./activations'):
    """CSV로 내보내기"""
    print("\n" + "=" * 80)
    print("10. EXPORTING TO CSV")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if df_stats is not None:
        stats_path = os.path.join(output_dir, 'layer_statistics.csv')
        df_stats.to_csv(stats_path, index=False)
        print(f"\n✓ Layer statistics saved: {stats_path}")
    
    if df_samples is not None:
        samples_path = os.path.join(output_dir, 'sample_summary.csv')
        df_samples.to_csv(samples_path, index=False)
        print(f"✓ Sample summary saved: {samples_path}")


def print_comprehensive_summary(activations_list, layer_stats):
    """종합 요약"""
    print("\n" + "=" * 80)
    print("11. COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    if len(activations_list) == 0:
        print("\n✗ WARNING: No samples were processed!")
        print("✗ This likely means all training samples had value=0 (Unknown)")
        print("✗ Please check your data filtering logic in process_pas()")
        print("\n" + "=" * 80)
        return
    
    print(f"\n✓ Total samples processed: {len(activations_list)}")
    print(f"✓ Total layers with interventions: {len(layer_stats)}")
    
    total_interventions = sum(sum(len(interventions) for interventions in sample['activations'].values()) 
                              for sample in activations_list)
    print(f"✓ Total interventions across all samples: {total_interventions}")
    print(f"✓ Average interventions per sample: {total_interventions / len(activations_list):.2f}")
    
    all_stds_global = []
    for stats in layer_stats.values():
        all_stds_global.extend(stats['std_values'])
    
    print(f"\n✓ Std value range: [{np.min(all_stds_global):.4f}, {np.max(all_stds_global):.4f}]")
    print(f"✓ Average std value: {np.mean(all_stds_global):.4f} ± {np.std(all_stds_global):.4f}")
    
    print(f"\n✓ Activations are ready for use in intervention experiments!")
    print("\n" + "=" * 80)


def main():
    """메인 함수"""
    # 파일 경로
    activations_path = './activations/PAS_Meta-Llama-3-8B-Instruct_OOD.pkl'
    output_dir = './activations'
    
    print("\n" + "🔍" * 40)
    print("PAS ACTIVATIONS ANALYSIS")
    print("🔍" * 40)
    
    # 1. 파일 로드
    activations_list = load_activations(activations_path)
    
    # 2. 데이터 구조 확인
    has_data = check_data_structure(activations_list)
    
    if not has_data:
        print("\n⚠️  Analysis stopped: No data available.")
        return
    
    # 3. Activations 상세 분석
    analyze_activations_detail(activations_list)
    
    # 4. 레이어별 통계
    layer_stats, df_stats = compute_layer_statistics(activations_list)
    
    # 5. Std 분포 시각화
    plot_std_distribution(layer_stats, output_dir)
    
    # 6. Direction Norm 분포 시각화
    plot_direction_norm_distribution(layer_stats, output_dir)
    
    # 7. 레이어별 헤드 수 시각화
    plot_heads_per_layer(layer_stats, output_dir)
    
    # 8. 샘플별 요약
    df_samples = compute_sample_summary(activations_list)
    
    # 9. Direction 유사도 분석
    analyze_direction_similarity(activations_list, output_dir)
    
    # 10. CSV 내보내기
    export_to_csv(df_stats, df_samples, output_dir)
    
    # 11. 종합 요약
    print_comprehensive_summary(activations_list, layer_stats)
    
    print("\n✅ Analysis completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
