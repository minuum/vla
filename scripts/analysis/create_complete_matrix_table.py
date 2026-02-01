#!/usr/bin/env python3
"""
Complete Experiment Matrix Table (CALVIN Style)
전체 16개 케이스를 논문 TABLE VII 스타일로 시각화

참고: CALVIN TABLE VII - Architecture, Train data, Task performance
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('docs/visualizations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("Complete Experiment Matrix Table Generator (CALVIN Style)")
print("="*80)
print()

# ============================================================================
# 전체 16개 케이스 정의
# ============================================================================

all_cases = {
    # L+R (500 episodes) + Chunk=10
    1: {
        'data': 'L+R (500)',
        'chunk': 10,
        'strategy': 'Baseline',
        'status': 'Completed',
        'val_loss': 0.027,
        'epochs': 10,
        'notes': 'Frozen+LoRA baseline'
    },
    2: {
        'data': 'L+R (500)',
        'chunk': 10,
        'strategy': 'Xavier Init',
        'status': 'Completed',
        'val_loss': 0.048,
        'epochs': 10,
        'notes': 'Performance degraded'
    },
    6: {
        'data': 'L+R (500)',
        'chunk': 10,
        'strategy': 'Abs Action',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': 'abs_action only'
    },
    3: {
        'data': 'L+R (500)',
        'chunk': 10,
        'strategy': 'Aug+Abs',
        'status': 'Completed',
        'val_loss': 0.050,
        'epochs': 10,
        'notes': 'Augmentation + abs'
    },
    
    # L+R (500 episodes) + Chunk=1 (No Chunk)
    5: {
        'data': 'L+R (500)',
        'chunk': 1,
        'strategy': 'Baseline',
        'status': 'Completed',
        'val_loss': 0.000532,
        'epochs': 7,
        'notes': '**BEST** No Chunk'
    },
    7: {
        'data': 'L+R (500)',
        'chunk': 1,
        'strategy': 'Xavier Init',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': 'Low priority'
    },
    8: {
        'data': 'L+R (500)',
        'chunk': 1,
        'strategy': 'Abs Action',
        'status': 'Completed',
        'val_loss': 0.00243,
        'epochs': 5,
        'notes': 'No Chunk + abs'
    },
    9: {
        'data': 'L+R (500)',
        'chunk': 1,
        'strategy': 'Aug+Abs',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': 'Recommended Tier 1'
    },
    
    # R only (250 episodes) + Chunk=10
    4: {
        'data': 'R only (250)',
        'chunk': 10,
        'strategy': 'Baseline',
        'status': 'Completed',
        'val_loss': 0.016,
        'epochs': 10,
        'notes': 'Right direction only'
    },
    10: {
        'data': 'R only (250)',
        'chunk': 10,
        'strategy': 'Xavier Init',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': 'Low priority'
    },
    11: {
        'data': 'R only (250)',
        'chunk': 10,
        'strategy': 'Abs Action',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': ''
    },
    12: {
        'data': 'R only (250)',
        'chunk': 10,
        'strategy': 'Aug+Abs',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': ''
    },
    
    # R only (250 episodes) + Chunk=1
    13: {
        'data': 'R only (250)',
        'chunk': 1,
        'strategy': 'Baseline',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': 'Reference only'
    },
    14: {
        'data': 'R only (250)',
        'chunk': 1,
        'strategy': 'Xavier Init',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': ''
    },
    15: {
        'data': 'R only (250)',
        'chunk': 1,
        'strategy': 'Abs Action',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': ''
    },
    16: {
        'data': 'R only (250)',
        'chunk': 1,
        'strategy': 'Aug+Abs',
        'status': 'Not Started',
        'val_loss': None,
        'epochs': None,
        'notes': ''
    },
}

# ============================================================================
# TABLE: Complete Experiment Matrix
# ============================================================================

def create_complete_matrix_table():
    """전체 실험 매트릭스 표 (CALVIN TABLE VII 스타일)"""
    
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # 데이터 준비
    table_data = []
    
    for case_id in sorted(all_cases.keys()):
        data = all_cases[case_id]
        
        # Val Loss 포맷
        if data['val_loss'] is not None:
            if data['val_loss'] < 0.001:
                val_loss_str = f"{data['val_loss']:.6f}"
            else:
                val_loss_str = f"{data['val_loss']:.3f}"
        else:
            val_loss_str = "-"
        
        # Epochs
        epochs_str = str(data['epochs']) if data['epochs'] else "-"
        
        # Status 아이콘
        if data['status'] == 'Completed':
            status_icon = '✓'
        elif  data['status'] == 'Not Started':
            status_icon = '○'
        else:
            status_icon = '~'
        
        table_data.append([
            f"Case {case_id}",
            data['data'],
            f"Chunk={data['chunk']}",
            data['strategy'],
            status_icon,
            val_loss_str,
            epochs_str,
            data['notes']
        ])
    
    # 헤더
    headers = ['Case', 'Data\n(episodes)', 'Action\nChunking', 'Strategy', 'Status', 
               'Val Loss', 'Epochs', 'Notes']
    
    # 테이블 생성
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.08, 0.12, 0.10, 0.12, 0.06, 0.10, 0.08, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # 헤더 스타일
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
    
    # Case별 색상 (완료된 케이스 강조)
    for i, case_id in enumerate(sorted(all_cases.keys())):
        row_idx = i + 1
        data = all_cases[case_id]
        
        if data['status'] == 'Completed':
            # 완료된 케이스
            for j in range(len(headers)):
                cell = table[(row_idx, j)]
                if case_id == 5:  # Best case
                    cell.set_facecolor('#d4edda')  # 녹색
                    if j in [0, 5]:  # Case ID와 Val Loss 강조
                        cell.set_text_props(weight='bold', fontsize=10)
                elif case_id == 8:  # 2nd best
                    cell.set_facecolor('#fff3cd')  # 노란색
                    if j == 5:
                        cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f8f9fa')  # 연한 회색
        elif 'Recommended' in data['notes']:
            # 권장 케이스
            for j in range(len(headers)):
                cell = table[(row_idx, j)]
                cell.set_facecolor('#e7f3ff')  # 연한 파란색
        
        # 그룹 구분선 (Data + Chunk 조합이 바뀔 때)
        if case_id in [5, 4, 13]:  # 그룹 경계
            for j in range(len(headers)):
                cell = table[(row_idx, j)]
                cell.set_linewidth(2)
                cell.set_edgecolor('black')
    
    # 제목
    title = ('TABLE II: Complete Experiment Matrix (16 Cases)\n' +
            'Mobile Navigation VLA - All Possible Configurations\n' +
            '✓ = Completed | ○ = Not Started | Data: Episodes | Chunk: fwd_pred_next_n')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=30, loc='center')
    
    # 범례
    legend_data = [
        ['Legend:', '', '', '', '', '', '', ''],
        ['✓', 'Training completed', '', '', '', '', '', ''],
        ['○', 'Not started (config not created)', '', '', '', '', '', ''],
        ['Green', 'Best performance (Case 5)', '', '', '', '', '', ''],
        ['Yellow', 'Runner-up (Case 8)', '', '', '', '', '', ''],
        ['Blue', 'Recommended for next experiments', '', '', '', '', '', ''],
    ]
    
    # 하단에 통계 추가
    stats_text = (
        f"\nStatistics:\n"
        f"• Total Cases: 16\n"
        f"• Completed: 6 (Cases 1, 2, 3, 4, 5, 8) - 37.5%\n"
        f"• Not Started: 10 - 62.5%\n"
        f"• Recommended Next: Cases 9 (Tier 1)\n"
        f"• Best Performance: Case 5 (Val Loss: 0.000532)\n"
    )
    
    plt.figtext(0.1, 0.02, stats_text, fontsize=10, 
                verticalalignment='bottom', family='monospace')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(OUTPUT_DIR / 'table2_complete_experiment_matrix.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Created: {OUTPUT_DIR / 'table2_complete_experiment_matrix.png'}")
    plt.close()

# ============================================================================
# Summary by Configuration Groups
# ============================================================================

def create_configuration_groups_table():
    """설정 그룹별 요약 표"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # 그룹별 요약
    groups = [
        {
            'name': 'L+R (500) + Chunk=10',
            'cases': [1, 2, 6, 3],
            'description': 'Full data with standard chunking'
        },
        {
            'name': 'L+R (500) + Chunk=1',
            'cases': [5, 7, 8, 9],
            'description': 'Full data with No Chunk strategy'
        },
        {
            'name': 'R only (250) + Chunk=10',
            'cases': [4, 10, 11, 12],
            'description': 'Reduced data with standard chunking'
        },
        {
            'name': 'R only (250) + Chunk=1',
            'cases': [13, 14, 15, 16],
            'description': 'Reduced data with No Chunk'
        },
    ]
    
    table_data = []
    for group in groups:
        # 그룹 정보
        cases_str = ', '.join([f"Case {c}" for c in group['cases']])
        
        # 완료된 케이스
        completed = [c for c in group['cases'] if all_cases[c]['status'] == 'Completed']
        completed_str = ', '.join([f"{c}" for c in completed]) if completed else "-"
        
        # 최고 성능
        best_val = None
        best_case = None
        for c in completed:
            if all_cases[c]['val_loss'] is not None:
                if best_val is None or all_cases[c]['val_loss'] < best_val:
                    best_val = all_cases[c]['val_loss']
                    best_case = c
        
        best_str = f"Case {best_case}: {best_val:.6f}" if best_case else "-"
        
        table_data.append([
            group['name'],
            group['description'],
            cases_str,
            f"{len(completed)}/4",
            completed_str,
            best_str
        ])
    
    headers = ['Configuration', 'Description', 'Cases', 'Completed', 'Completed Cases', 'Best Performance']
    
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.20, 0.25, 0.15, 0.10, 0.12, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # 헤더 스타일
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white')
    
    # No Chunk 그룹 강조
    for j in range(len(headers)):
        cell = table[(2, j)]  # L+R + Chunk=1
        cell.set_facecolor('#e7f3ff')
    
    ax.set_title('TABLE III: Configuration Groups Summary\n' +
                'Experiment Progress by Data and Chunking Strategy',
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'table3_configuration_groups.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Created: {OUTPUT_DIR / 'table3_configuration_groups.png'}")
    plt.close()

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Creating complete experiment matrix tables...")
    print()
    
    print("[1/2] Complete Experiment Matrix Table...")
    create_complete_matrix_table()
    print()
    
    print("[2/2] Configuration Groups Summary...")
    create_configuration_groups_table()
    print()
    
    print("="*80)
    print("✓ All tables created successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*80)
