"""
probe_trainer.py
================
Probe 학습 및 Direction 탐색

논문 4.2절:
- 각 head에 대해 LogisticRegression probe 학습
- 60% train / 40% validation split
- Validation accuracy 기준 Top-K head 선택
- 선택된 head의 정규화된 probe weights가 direction
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from config import (
    DEFAULT_TRAIN_RATIO, DEFAULT_NUM_HEADS,
    PROBE_MAX_ITER, PROBE_RANDOM_STATE, NORM_EPS
)


@dataclass
class ProbeResult:
    """단일 head의 probe 학습 결과"""
    layer: int
    head: int
    train_acc: float
    val_acc: float
    direction: np.ndarray  # 정규화된 probe weights (shape: head_dim)
    probe: LogisticRegression


@dataclass
class InterventionHead:
    """Intervention 대상 head 정보"""
    layer: int
    head: int
    direction: np.ndarray  # 정규화된 direction vector
    sigma: float           # direction 방향 activation의 std
    val_acc: float         # validation accuracy


class ProbeTrainer:
    """
    Head별 probe 학습 및 direction 탐색

    논문 방식:
    1. agree/disagree activation 쌍에서 차이 계산
    2. 각 head에 대해 probe (LogisticRegression) 학습
    3. Validation accuracy로 head 랭킹
    4. Top-K head의 정규화된 weights를 direction으로 사용
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        random_state: int = PROBE_RANDOM_STATE
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.train_ratio = train_ratio
        self.random_state = random_state

        # 결과 저장
        self.probe_results: List[ProbeResult] = []
        self.top_heads: List[InterventionHead] = []

    def prepare_training_data(
        self,
        agree_activations: np.ndarray,
        disagree_activations: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Probe 학습용 데이터 준비

        논문 방식: agree와 disagree의 activation을 각각 positive/negative sample로 사용
        label=1이면 agree가 positive 방향, label=0이면 disagree가 positive 방향

        Args:
            agree_activations: (n_samples, num_layers, num_heads, head_dim)
            disagree_activations: (n_samples, num_layers, num_heads, head_dim)
            labels: (n_samples,) - 1 or 0

        Returns:
            X: (2*n_samples, num_layers, num_heads, head_dim) - 모든 activation
            y: (2*n_samples,) - 0 (negative direction) or 1 (positive direction)
        """
        n_samples = len(labels)

        # 각 sample에 대해 agree와 disagree를 적절한 label로 배치
        X_list = []
        y_list = []

        for i in range(n_samples):
            # Agree activation
            X_list.append(agree_activations[i])
            # label=1이면 agree가 positive (y=1), label=0이면 agree가 negative (y=0)
            y_list.append(labels[i])

            # Disagree activation
            X_list.append(disagree_activations[i])
            # label=1이면 disagree가 negative (y=0), label=0이면 disagree가 positive (y=1)
            y_list.append(1 - labels[i])

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y

    def train_probes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        show_progress: bool = True
    ) -> List[ProbeResult]:
        """
        모든 head에 대해 probe 학습

        Args:
            X: (n_samples, num_layers, num_heads, head_dim)
            y: (n_samples,)
            show_progress: tqdm 표시 여부

        Returns:
            ProbeResult 리스트 (num_layers * num_heads 개)
        """
        n_samples = X.shape[0]

        # Train/validation split
        np.random.seed(self.random_state)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        train_size = int(n_samples * self.train_ratio)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        print(f"Training probes: {len(train_idx)} train, {len(val_idx)} validation samples")

        # Label 분포 확인
        y_train = y[train_idx]
        y_val = y[val_idx]
        print(f"  Train labels: {(y_train==1).sum()} pos, {(y_train==0).sum()} neg")
        print(f"  Val labels: {(y_val==1).sum()} pos, {(y_val==0).sum()} neg")

        self.probe_results = []

        total_heads = self.num_layers * self.num_heads
        iterator = range(total_heads)
        if show_progress:
            iterator = tqdm(iterator, desc="Training probes")

        for flat_idx in iterator:
            layer = flat_idx // self.num_heads
            head = flat_idx % self.num_heads

            # 해당 head의 activation 추출
            X_head = X[:, layer, head, :]  # (n_samples, head_dim)

            X_train = X_head[train_idx]
            X_val = X_head[val_idx]

            # NaN/inf 처리
            if not np.isfinite(X_train).all():
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(X_val).all():
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

            # Probe 학습
            try:
                probe = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=PROBE_MAX_ITER,
                    solver='lbfgs'
                )
                probe.fit(X_train, y_train)

                # 정확도 계산
                train_acc = accuracy_score(y_train, probe.predict(X_train))
                val_acc = accuracy_score(y_val, probe.predict(X_val))

                # Direction 추출 및 정규화
                direction = probe.coef_[0]  # shape: (head_dim,)
                norm = np.linalg.norm(direction)
                if norm > NORM_EPS:
                    direction = direction / norm
                else:
                    direction = np.zeros_like(direction)

            except Exception as e:
                print(f"  Warning: Probe training failed for layer {layer}, head {head}: {e}")
                probe = None
                train_acc = 0.5
                val_acc = 0.5
                direction = np.zeros(self.head_dim)

            result = ProbeResult(
                layer=layer,
                head=head,
                train_acc=train_acc,
                val_acc=val_acc,
                direction=direction,
                probe=probe
            )
            self.probe_results.append(result)

        return self.probe_results

    def select_top_heads(
        self,
        X: np.ndarray,
        num_heads: int = DEFAULT_NUM_HEADS
    ) -> List[InterventionHead]:
        """
        Validation accuracy 기준 Top-K head 선택

        Args:
            X: activation data (sigma 계산용)
            num_heads: 선택할 head 수 (K)

        Returns:
            InterventionHead 리스트
        """
        if not self.probe_results:
            raise ValueError("No probe results. Call train_probes() first.")

        # Validation accuracy로 정렬
        sorted_results = sorted(
            self.probe_results,
            key=lambda r: r.val_acc,
            reverse=True
        )

        # Top-K 선택
        top_k = sorted_results[:num_heads]

        print(f"\nTop {num_heads} heads by validation accuracy:")
        print(f"  Accuracy range: {top_k[-1].val_acc:.4f} ~ {top_k[0].val_acc:.4f}")

        self.top_heads = []

        for result in top_k:
            # Sigma 계산: direction 방향으로 projection한 값들의 std
            X_head = X[:, result.layer, result.head, :]  # (n_samples, head_dim)

            # Direction 방향 projection
            projections = X_head @ result.direction  # (n_samples,)
            sigma = np.std(projections)

            if not np.isfinite(sigma):
                sigma = 1.0  # 기본값

            intervention_head = InterventionHead(
                layer=result.layer,
                head=result.head,
                direction=result.direction,
                sigma=sigma,
                val_acc=result.val_acc
            )
            self.top_heads.append(intervention_head)

        return self.top_heads

    def get_intervention_dict(self) -> Dict[str, List[Tuple[int, np.ndarray, float]]]:
        """
        Steering에 사용할 intervention 딕셔너리 반환

        Returns:
            {layer_name: [(head_idx, direction, sigma), ...]}
        """
        if not self.top_heads:
            raise ValueError("No top heads selected. Call select_top_heads() first.")

        interventions = {}

        for head in self.top_heads:
            layer_name = f"model.layers.{head.layer}.self_attn.head_out"

            if layer_name not in interventions:
                interventions[layer_name] = []

            interventions[layer_name].append((
                head.head,
                head.direction,
                head.sigma
            ))

        # Head index로 정렬
        for layer_name in interventions:
            interventions[layer_name] = sorted(
                interventions[layer_name],
                key=lambda x: x[0]
            )

        return interventions

    def get_accuracy_stats(self) -> Dict[str, float]:
        """전체 probe accuracy 통계"""
        if not self.probe_results:
            return {}

        val_accs = [r.val_acc for r in self.probe_results]
        train_accs = [r.train_acc for r in self.probe_results]

        return {
            'mean_val_acc': np.mean(val_accs),
            'std_val_acc': np.std(val_accs),
            'max_val_acc': np.max(val_accs),
            'min_val_acc': np.min(val_accs),
            'mean_train_acc': np.mean(train_accs),
            'above_random': np.sum(np.array(val_accs) > 0.55) / len(val_accs)
        }


def train_and_select_heads(
    agree_activations: np.ndarray,
    disagree_activations: np.ndarray,
    labels: np.ndarray,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_to_select: int = DEFAULT_NUM_HEADS,
    train_ratio: float = DEFAULT_TRAIN_RATIO
) -> Tuple[ProbeTrainer, Dict]:
    """
    전체 파이프라인: probe 학습 → top head 선택 → intervention dict 생성

    Returns:
        (trainer, intervention_dict)
    """
    trainer = ProbeTrainer(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        train_ratio=train_ratio
    )

    # 데이터 준비
    X, y = trainer.prepare_training_data(agree_activations, disagree_activations, labels)
    print(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")

    # Probe 학습
    trainer.train_probes(X, y)

    # 통계 출력
    stats = trainer.get_accuracy_stats()
    print(f"\nProbe accuracy statistics:")
    print(f"  Mean val acc: {stats['mean_val_acc']:.4f} ± {stats['std_val_acc']:.4f}")
    print(f"  Range: [{stats['min_val_acc']:.4f}, {stats['max_val_acc']:.4f}]")
    print(f"  Above random (>0.55): {stats['above_random']*100:.1f}%")

    # Top head 선택
    trainer.select_top_heads(X, num_to_select)

    # Intervention dict 생성
    intervention_dict = trainer.get_intervention_dict()
    print(f"\nIntervention layers: {len(intervention_dict)}")
    for layer_name, heads in intervention_dict.items():
        print(f"  {layer_name}: {len(heads)} heads")

    return trainer, intervention_dict


if __name__ == "__main__":
    print("ProbeTrainer module loaded.")
    print("Usage: train_and_select_heads(agree_acts, disagree_acts, labels, ...)")
