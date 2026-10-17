# PREREG_35 — U6 프로브 P1 / P3 사전등록 (동결 시각 2026-10-15 12:13, 실행자: PROBE RUNNER; 후보 저자·감사자·재작성자 아님)

**동결 규칙**: 이 파일은 결과를 보기 전에 작성되었고 이후 수정하지 않는다(버그 정정은 말미에 "AMENDMENT"로만 추가). 근거 허용 범위: S-c(P3) 실행 결과만. 아래 인용은 kill 근거가 아님.

## 0. 근거 인용 상태 (kill 근거 아님)
- Nagel et al., arXiv 2106.08295 §3.7 — **미확인 (원문 대조 전)**. kill 근거에서 제외.
- Pandey et al., arXiv 2302.05397 §3.2 — **미확인 (원문 대조 전)**. kill 근거에서 제외.
- 판정 근거는 오직 이 문서의 P3(및 보조 P1) 실행 결과.

## 1. 적용 조건 (9101–9103·9601–9602 기록과 동일; 재사용한 코드 file:line)
기록 코드 위치: `results/loop/inbox/9101_gpu_precision_weight_rtn/precision_probe.py` (9102·9103과 sha256 동일 178558633d3d…c331), `.../9601_gpu_prune_skin/prune_probe.py` (9602와 sha256 동일 e930169c65dd…c8df). 아래는 이를 `p3_agree.py`에 그대로 복사한 연산.
- **w4_pt** = per-tensor symmetric RTN, 4 bit, scale = max|w|/7, levels −7..7, 대상 = 모든 float 파라미터 dim≥2 (BN 미접힘): `precision_probe.py:96-101`(rtn_sym_tensor), 대상 선정 `:485`, 적용 `:511`.
- **a6** = 모든 Conv/ConvTranspose/Linear 모듈 출력(pre-BN)에 per-tensor **동적** 대칭 fake-quant, 6 bit(qm=31, amax는 해당 forward 출력 텐서, batch 1), 가중치는 fp32: `precision_probe.py:492-493`(hook 대상), `:525-537`(hook), batch 1 `eval_config :133-`.
- **l50** = layer-wise unstructured magnitude pruning 50%: k=floor(0.5·n), |w| ≤ k번째 최소값을 0으로, 대상 = forward에서 실제 호출된 Conv/Linear의 float dim≥2 weight(Parameter 중복 제거): `prune_probe.py:87-103`(prunable_layers), `:106-127`(kth/prune_layer), 활성 판정 `:386-396`.
- 마스크 = `sigmoid(logit) > 0.5` (`precision_probe.py:153`). 입력 해상도 256×256, `uint8/255` float32 (SegCache와 동일).
- 모델: `code/pbe_seg` (registry, 11모델 DEFAULT_CFG), strict load. 로컬 repo는 HEAD 98f478d이며 기록 시점 fbe35ca와 11개 모델 디렉터리 diff 없음(lce/diag_view 추가·registry 1줄만 다름). 66개 체크포인트 파일의 sha256이 9901 `results.json`의 기록 해시와 **66/66 일치**(사전 확인, 점수와 무관).
- 체크포인트: 10 baseline = `<workspace>/fieldswitch/ckpt/<model>/<ds>/k1/checkpoint.pth`, PBE = `<workspace>/iso/<WORK>/results/queue10_pbe_reference_k1_v1/data/<ds>/checkpoint.pth`.
- 영상: `<workspace>/data/cache/v1/<ds>/test_1_*.npz`의 `images` 키 앞 8장 (저장 순서, 선택 없음). **`masks` 키는 읽지 않음.** (데이터셋 6: etis, PH2, cvc, kvasir, isic17, isic18)
- 환경: CPU, torch 2.4.1+cpu, 2 스레드, 프로세스 1개, GPU/서버/[internal review]/다운로드 0, 기존 파일 수정·삭제 0. 가중치 복원 byte-exact 검사 포함.
- 환경 우회(수학에 영향 없음, 파일 수정 없음): 로컬에 `torchsummary` 없음 + matplotlib NumPy ABI 불일치 → 해당 import를 프로세스 내에서만 stub(`stub_unavailable_imports`). forward 수치와 무관.

**알려진 조건 불일치(결과 해석에 반영)**: 기록의 retention은 **val_1** 분할 전체의 평균 IoU에서 나왔고(`val_1_*.npz`는 로컬에 없음), P3 점수는 지시에 따라 **test_1** 앞 8장에서 계산한다. 즉 점수와 정답은 서로 다른 영상 집합(같은 체크포인트·같은 연산)이다. 기록은 GPU, P3는 CPU(반올림 수준 차이 가능).

## 2. 정답(붕괴) 정의 — 기록에서 가져옴, P3 점수를 보기 전 확정
- 출처: `loop/inbox/9901_cpu_quantz_stats/observed_collapse.json` (sha256 f7e9fca2947549235fe4df2b037f4de489900c40d21eb93932de6922d0c4dd8d, 9101/9102/9601/9602의 results.json에서 2026-10-14 동결본). 셀 = (모델 11 × 데이터셋 6) × cfg {w4_pt, a6, l50} = 66 × 3 = **198 셀**.
- **이진 붕괴(binary collapse)**: 기록 함수 `collapsed(orig, q, chance_max)`(`precision_probe.py:176-182`, `prune_probe.py:207`) 그대로: orig ≤ chance+0.05 이면 False(이미 chance), 아니면 q ≤ chance+0.05 이면 True. `chance` = 기록의 chance(= max(all-bg IoU, all-fg IoU)). 이 파일에서 orig≤chance+0.05 셀은 0개(사전 계수: 붕괴 w4_pt 3 / a6 8 / l50 35 of 66, 합 46 of 198).
- **연속 타깃**: retention = IoU(cfg)/IoU(orig) (REWRITE_35 §2 정의). "drop" = 1 − retention이므로 Spearman(score, retention) = −Spearman(score, drop).

## 3. 점수
### P3 (S-c, 라벨 불필요)
- 체크포인트별·cfg별: 이미지 i에서 A_i = (sigmoid(logit_fp32) > 0.5), B_i = (sigmoid(logit_q) > 0.5), Dice_i = 2|A_i∩B_i| / (|A_i|+|B_i|).
- **퇴화 규칙**: |A_i|+|B_i| = 0 (둘 다 빈 마스크) → Dice_i = 1 (기록의 IoU u=0→1 규칙과 동일). 한쪽만 비면 0. NaN/Inf logit은 마스크 False로 처리(별도 카운트 없음).
- **score(체크포인트, cfg) = mean_{i=1..8} Dice_i.** 방향: **클수록 건강**(retention과 양의 상관 기대). 붕괴 탐지 AUC는 "score가 작으면 붕괴"로 정의: AUC = P(score_healthy > score_collapsed) + 0.5·P(동점).
### P1 (가중치 SQNR, 추론 없음)
- SQNR_dB(W, Wq) = 10·log10(‖W‖² / ‖W−Wq‖²) (float64). 오차 0이면 100 dB 상한, ‖W‖=0 텐서는 제외.
- **score_w4(체크포인트)** = min over 대상 텐서(모든 float 파라미터 dim≥2) SQNR_dB(W, RTN_w4_per-tensor(W)).
- **score_l50(체크포인트)** = min over Conv/Linear 모듈 weight(float dim≥2, 정적 — 추론 없이 호출 여부 무시) SQNR_dB(W, layerwise_prune50(W)) (= 제거된 가중치 에너지 대비 잔존 에너지).
- 모델 수준 값 = 6 데이터셋 체크포인트 점수의 평균, 타깃 = 같은 6개 retention의 평균 (9901 방식). a6용 가중치-only 공식은 없음 → a6 vs score_w4는 **기술(descriptive)만**, 판정 제외.

## 4. 판정 기준 (조정자 고정; 풀링 선택은 여기서 확정)
**P3 PASS ⇔ (A) 세 cfg 각각에서 66 셀 Spearman ρ(score, retention) ≥ 0.7 (3개 모두), 그리고 (B) 198 셀 풀링 이진 붕괴 AUC ≥ 0.9.** 하나라도 미달이면 FAIL. 보고만 하는 값: 풀링 ρ, cfg별 AUC, 붕괴 클래스 수(w4_pt는 3개뿐이라 cfg별 AUC는 참고용). 198 셀이 모두 채워지지 않으면 판정은 `PARTIAL`이며 부분 수를 그대로 보고(설계 변경 금지).
**P1 PASS ⇔ |ρ| ≥ 0.7 이 다음 4개 모두에서 성립**: (score_w4 ↔ w4_pt retention, 11모델 수준), (같음, 66셀 수준), (score_l50 ↔ l50 retention, 11모델), (같음, 66셀). (a6은 제외.)
**분기**: P3 PASS → U6 paper kill 확정(S-c 확인). P3 FAIL → U6 측정 후보로 재개(pred_kill 0.85, cost_h 0.5). P1은 보조 — P1 PASS는 "체크포인트-only 예측자 존재"를 뜻하나 kill 분기를 바꾸지 않고 별도 기록.
**시간 규칙**: P1 ≤ 0.1 h, P3 ≤ 0.5 h(스크립트 wall guard 1800 s; 초과 예상 시 중단하고 부분 수 보고).

## 5. 스크립트와 sha256 (작성 후·실데이터 점수 산출 전 계산; 시각 2026-10-15 12:12:48)
| 파일 | sha256 |
|---|---|
| `AUDIT/p3_agree.py` | `23808f6e61472752aba97bc46d20858bf2a5513bca206f7f21836646d140d045` |
| `AUDIT/p1_sqnr.py` | `d3ac4672444af09daf810cbc8df413e2954c543bb3f0a39fd35cb36264872c60` |
(p1_sqnr.py는 p3_agree.py의 spearman/auc/연산을 import — 두 해시가 모두 동결 대상.)
**투명성 고지**: 해시 전에 (1) 모델별 forward 시간만 재는 타이밍 스크립트(<scratch>, 점수 없음), (2) `--smoke` 1회(lbunet/etis 1개 체크포인트, 출력 = wall·sha 일치·복원 정합·적용 층 수뿐, 점수 미출력)를 실행했다. smoke 출력: `sha_matches_9901=True state_restored=True integrity={'w4_pt': {n_target_tensors 104, max_unique_levels 15}, 'a6': {n_hooked_modules 80}, 'l50': {n_pruned_layers 80, min_zero_frac 0.5}}`(80 = 9901의 lbunet active layer 수와 일치).

## 6. 합성 역검사 (SYNTHETIC reverse-check; 실데이터 아님, 동일 `analyze()` 함수)
```
SYNTHETIC perfect_monotone(score=retention^3): pooled rho=1.0000 AUC=1.0000 per-cfg rho=1.000,1.000,1.000 verdict=PASS
SYNTHETIC anti_correlated(score=-retention): pooled rho=-1.0000 AUC=0.0000 per-cfg rho=-1.000,-1.000,-1.000 verdict=FAIL
SYNTHETIC random_noise: pooled rho=0.0345 AUC=0.5148 per-cfg rho=-0.194,0.159,0.145 verdict=FAIL
SYNTHETIC sign-check: rho(score=+ret)=1.0000, rho(score=-ret)=-1.0000, AUC(+)=1.0000, AUC(-)=0.0000 (1-AUC(-)=1.0000)
SYNTHETIC tie-check: rho([1,1,2,2],[5,5,9,9])=1.0000 (expect 1), rho([1,1,2,2],[9,9,5,5])=-1.0000 (expect -1), rho([1,2,3,4],[1,1,1,1])=None (expect None)
SYNTHETIC auc all-equal scores: 0.500 (expect 0.500)
SYNTHETIC P1 perfect-monotone: rho_cell(w4_pt)=1.000 rho_model(w4_pt)=1.000 rho_cell(l50)=1.000 rho_model(l50)=1.000 verdict=PASS
SYNTHETIC P1 anti-monotone (|rho| rule): rho_cell(w4_pt)=-1.000 rho_model(w4_pt)=-1.000 verdict=PASS
```
부호 규약: 완전 반상관 score(−retention)은 ρ=−1, AUC=0 → 부호 반전 후 ρ=+1, AUC=1. 실데이터에서는 부호를 사후에 뒤집지 않는다(음의 ρ는 FAIL). scipy는 로컬 NumPy ABI 불일치로 사용 불가 → 손으로 검증 가능한 동점 사례로 대체.

## 7. 결과를 보기 전 기록하는 위험·한계
- 점수(test 8장)와 정답(val 전체)이 다른 영상 집합이라 소표본 잡음이 ρ를 깎을 수 있음(8장은 지시 고정).
- l50은 66 셀 중 35가 붕괴 → 동점이 많은 retention(0 근처) 구간에서 ρ는 순위 동점 영향을 받음(평균 순위 처리).
- fp32가 거의 빈/가득 마스크인 체크포인트(예: etis)에서는 Dice가 붕괴와 무관하게 치우칠 수 있음.
- 이 프로브는 "측정"이며 사전 예측 보증이 아니다(양자화 추론 8장 필요).
