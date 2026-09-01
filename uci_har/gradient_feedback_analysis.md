# Iterative gradient feedback

동일한 초기화와 동일한 128개 sample을 사용해 one-shot, frozen-gradient sequential, recomputed sequential을 비교했다. test는 update에 사용하지 않았다.

## Final summary
- one_shot: movement=0.195, test accuracy=0.279, test separation=0.511, val/test cosine=0.976, final direction cosine=0.696
- frozen: movement=5.940, test accuracy=0.337, test separation=3.500, val/test cosine=0.995, final direction cosine=0.882
- recomputed: movement=2.016, test accuracy=0.706, test separation=2.419, val/test cosine=0.993, final direction cosine=0.380

## 판정

Recomputed가 frozen/one-shot보다 test geometry와 accuracy에서 우수하고, gradient cosine이 update 중 변하면 iterative feedback 가설을 지지한다. 다만 raw 결과는 update 횟수와 movement 차이를 포함하므로 same-norm 결과를 함께 본다.

`current representation → current error → gradient → parameter change → new representation → new gradient → trajectory → unseen generalization`
