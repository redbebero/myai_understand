# Hidden representation compression

학습된 561→64→32→6 MLP의 마지막 hidden representation(32차원)을 train 표현에서 계산한 basis로 압축했다. 압축 후 32차원으로 복원하고 출력층은 고정했다. test는 basis 학습에 사용하지 않았다.

원본 test accuracy: 0.938
원본 prediction 기준: 1.000

| 방법 | k | accuracy | prediction agreement | distance correlation | separation ratio |
|---|---:|---:|---:|---:|---:|
| random_neuron | 32 | 0.938 | 1.000 | 1.000 | 3.761 |
| random_neuron | 16 | 0.846 | 0.868 | 0.942 | 3.639 |
| random_neuron | 8 | 0.609 | 0.618 | 0.867 | 3.868 |
| random_neuron | 4 | 0.416 | 0.417 | 0.709 | 3.881 |
| random_neuron | 2 | 0.296 | 0.294 | 0.547 | 4.033 |
| random_neuron | 1 | 0.231 | 0.227 | 0.401 | 3.893 |
| random_orthogonal | 32 | 0.938 | 1.000 | 1.000 | 3.761 |
| random_orthogonal | 16 | 0.849 | 0.870 | 0.959 | 3.741 |
| random_orthogonal | 8 | 0.619 | 0.627 | 0.892 | 3.901 |
| random_orthogonal | 4 | 0.413 | 0.413 | 0.774 | 3.644 |
| random_orthogonal | 2 | 0.306 | 0.302 | 0.617 | 3.750 |
| random_orthogonal | 1 | 0.243 | 0.239 | 0.504 | 4.206 |
| pca | 32 | 0.938 | 1.000 | 1.000 | 3.761 |
| pca | 16 | 0.937 | 0.994 | 1.000 | 4.011 |
| pca | 8 | 0.938 | 0.991 | 0.999 | 4.563 |
| pca | 4 | 0.898 | 0.928 | 0.987 | 6.762 |
| pca | 2 | 0.723 | 0.733 | 0.933 | 9.097 |
| pca | 1 | 0.310 | 0.292 | 0.835 | 14.157 |
| class_separating | 32 | 0.938 | 1.000 | 1.000 | 3.761 |
| class_separating | 16 | 0.939 | 0.993 | 1.000 | 4.079 |
| class_separating | 8 | 0.937 | 0.989 | 0.998 | 4.802 |
| class_separating | 4 | 0.898 | 0.926 | 0.985 | 7.186 |
| class_separating | 2 | 0.740 | 0.744 | 0.935 | 9.584 |
| class_separating | 1 | 0.322 | 0.302 | 0.829 | 14.337 |
| supervised_output | 32 | 0.938 | 1.000 | 1.000 | 3.761 |
| supervised_output | 16 | 0.938 | 1.000 | 1.000 | 4.170 |
| supervised_output | 8 | 0.938 | 1.000 | 0.991 | 6.006 |
| supervised_output | 4 | 0.875 | 0.913 | 0.780 | 4.660 |
| supervised_output | 2 | 0.564 | 0.581 | 0.580 | 4.033 |
| supervised_output | 1 | 0.293 | 0.313 | 0.398 | 2.918 |

## 최소 차원

- random_neuron: 32
- random_orthogonal: 32
- pca: 8
- class_separating: 8
- supervised_output: 8

## 판정

class-separating 또는 supervised-output 방식이 같은 k에서 random/PCA보다 높은 정확도와 prediction agreement를 유지하면 관계 보존 가설을 지지한다. PCA가 우세하면 전체 분산 구조가 더 중요하다는 뜻이고, 모든 방식이 빠르게 무너지면 고정 출력층과 원래 좌표 정렬이 중요하다는 뜻이다.
