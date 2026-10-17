# PREREG_THEORYQ — 숨은 가지치기 명제와 가중치만의 붕괴 예측 사전등록 (**동결본**)

작성 2026-10-17. 상태: **동결**. 동결 시각과 sha는 서버 `<WORK>/theoryq/FROZEN_SHA256SUMS`·`FROZEN_AT`에 있다. 사본은 `prereg/original/09_PREREG_THEORYQ.md`다.
- 동결 시점까지 실제 checkpoint에서 예측기(Z, Z′, Zw, Mstat, E 등) 값은 **0회** 계산했다.
- 동결 전에 한 일은 셋이다. 모두 예측기 값을 내지 않는다.
  - (1) CPU 합성 역검사. 가짜 텐서만 썼다(§6).
  - (2) 구조 dry run 11 checkpoint(모델마다 etis k1). 그래프를 구성하고, F3 구조 필드와 대조하고, 제외 텐서 수와 시간만 냈다(§7.3).
  - (3) decide 코드 경로 자기검사. 무작위 가짜 예측기를 썼고 결과는 의미가 없다(서버 `theoryq/selftest_decide/`, 보존).
- 코디네이터 결정(2026-10-17)을 반영했다.
  - (1) PASS = 셀 부트스트랩 CI 하한 ≥ 문턱
  - (2) 실행되지 않은 텐서는 Z에서 제외하고, "무해 0"을 뺀 Z′는 보조로 둔다
  - (3) Zw는 보조다
  - (4) 서버 CPU에서 실행한다
  - (5) SGR-E 목적은 Σ√m_l이다
  - Δ는 코드 값을 쓴다
- 근거: `DESIGN_THEORY_Q.md`. F3 동결 코드는 셋이다: 서버 `qdeploy/qdeploy.py`(sha256 644bd5ee…4d96), `qdeploy/qdeploy_analyze.py`(f93895df…b899), `qfix/C/rec/qfix_a.py`(b36df82f…94c6). F3 결과는 `RESULT_QDEPLOY.json`, `qdeploy_tau/Q_ROBUST.json`이다.

---

## 0. 공개 사항

**이 검증은 맹검 예측이 아니다.** 66셀 × 모든 설정의 붕괴 라벨은 F3에서 이미 정해졌다(2026-10-16 decide). 작성자는 그 수를 안다.
- 알려진 수: S_w4g 38/66, O_w4 25/66, D_w4 9/66, D_l50 55/66.
- O_w4 붕괴 25셀은 모두 S_w4g에서도 붕괴한다. 새 붕괴 13셀은 모두 접기·융합이 있는 모델의 셀이다.
- D_l50 비붕괴 11셀은 malunet 3, maresunet 5, mhorunet 2, mk_unet 1이다.

안전장치는 하나다. **예측기 정의, 그래프·텐서 선택, 셀 통계, 문턱, AUC·PASS 규칙, 부트스트랩, Holm을 실제 가중치로 예측기를 계산하기 전에 동결한다.**
- 동결 뒤에는 예측기를 한 번만 계산하고, 판정을 한 번만 한다.
- 값을 본 뒤의 변경은 사후 탐색으로만 보고한다.
- 라벨을 알고 정의를 골랐다는 편향은 남는다. 원고에도 그대로 쓴다.

---

## 1. 실제 양자화 코드

`qfix_a.rtn_sym_tensor` (w4_pt, 설정 O_w4·S_w4g):
- qm = 2^(b−1) − 1 = 7, A = max|w|(텐서)
- s = A/qm(A가 유한하고 > 0일 때, 아니면 1)
- Q(w) = clamp(round(w/s), −qm, qm)·s

`qdeploy.rtn_sym_channel` (w4_pc, D_w4·S_w4c)은 같은 식을 출력 채널마다 적용한다. 채널 축은 ConvTranspose `.weight`이면 1이고, 그 밖은 0이다.

- 수준은 −7…7의 15개다. **눈금 Δ = A/7**이다.
- `torch.round`가 half-to-even이므로 **w ↦ 0 ⇔ |w| ≤ Δ/2 = A/14**다.
- clamp는 0을 만들지 않는다. bias(1차원)는 양자화되지 않는다.
- **설계와의 불일치.** 설계 §1은 Δ = 2A/(2^b − 1)로 썼고, 그러면 문턱은 A/15다. **코드 값(Δ = A/7, 문턱 A/14)을 쓴다.**
  - 차이가 나는 범위는 r_c/A ∈ (1/15, 1/14]인 채널이다. 합성 S2가 이를 보인다.

---

## 2. 명제

기호: W_l, 출력 채널 조각 W_{l,c}, r_{l,c} = max|W_{l,c}|, A_l = max_c r_{l,c}.

**Prop 1 (텐서별 양자화는 채널을 통째로 가지치기한다).** r_{l,c} > 0일 때 Q(W_l)_{l,c} ≡ 0 ⇔ r_{l,c} ≤ A_l/(2qm)이다(b = 4이면 A_l/14).
- 증명. (⇐) 채널의 모든 |w| ≤ s/2이므로 round = 0이다. (⇒) 최대 원소의 |w*/s| > 1/2이므로 |round| ≥ 1이고, clamp 뒤에도 그렇다. ∎
- 그 채널의 출력은 입력과 무관한 상수(bias, 접힌 BN shift)가 된다.
- **BN 접기**는 W'_{l,c} = t_c W_{l,c}, t_c = γ_c/√(σ²_c + ε)이다. 그러면 범위비 κ'_l ∈ [κ_l/τ_l, κ_l τ_l]이고, τ_l = max|t|/min|t|이다.
  - 접기는 범위비를 최대 τ_l배까지 넓힐 수 있다. 단조성은 정리가 아니다.
- **무해 0.** |γ_c| ≈ 0(BN이 끈 채널)이면 접은 뒤 0이 되어 Z에 잡힌다. 그러나 그 채널은 fp32에서도 이미 상수였다. 그래서 Z′(§3)로 따로 본다.

**Prop 2 (채널별 눈금).** `rtn_sym_channel`에서 r_{l,c} > 0이고 유한한 채널은 0이 되지 않는다.
- 증명: s_c = r_{l,c}/qm이므로 최대 원소의 수준은 ±qm이다. ∎
- 따라서 Z_pc ≡ 0이다(처음부터 0인 채널은 세지 않는다).

**Prop 3 (가지치기 섭동 상한).**
- 가정: bias 없는 DAG. 구성 요소는 선형 층, φ(0) = 0이고 1-Lipschitz인 활성, 덧셈·concat 병합, 노름이 상수인 고정 선형 연산이다.
- 층 노름 상한 N_l = √(k_h k_w)‖W_l‖_F, 경로합 상수 P_out = Σ_π Π N_j·Π c_op로 둔다.
- 그러면 **‖f(x) − f̃(x)‖ ≤ Σ_l G_l‖W_l ⊙ (1 − M_l)‖_F‖x‖, G_l = P_out/‖W_l‖_F**이다.
- 정규화하면 **‖f − f̃‖/(P_out‖x‖) ≤ E = Σ_l √m_l**, m_l = ‖W_l ⊙ (1 − M_l)‖²_F/‖W_l‖²_F이다.
- 증명은 노드 귀납이다. 가지치기는 Frobenius 노름을 줄인다는 점, √(kk)‖ΔW_l‖_F = N_l√m_l라는 점, 병합에서 D ≤ ΣD라는 점을 쓴다.
- **G_l을 이렇게 정한 이유.** 가중치만으로 계산된다. P_out은 모든 층에 공통이라 정규화에서 빠진다. 스펙트럼 곱은 깊이에서 폭발하고 U-Net skip에서 정의되지 않는다. √Mstat ≤ E ≤ L√Mstat이다.
- **한계.**
  - 상한은 매우 느슨하다(toy DAG에서 1.1e-5배, §6 S11).
  - 실제 11개 모델에는 bias, GELU, GN·LN, 곱셈 게이트·어텐션이 있다. 따라서 **Prop 3은 실제 모델에 대해 증명된 상한이 아니라 순위 근거일 뿐이다.**
  - H_T3은 "Prop 3이 동기를 준 통계가 붕괴 순위를 맞추는가"의 검증이다.

---

## 3. 예측기 (가중치만, 영상 0, 라벨 0)

### 3.1 그래프와 텐서
- **배포 그래프**: `qdeploy.make_dep`. PBE는 `f2_fuse.fuse_pbe`로, 나머지는 `FOLD_PAIRS.json` 쌍을 `fold_named`로 접는다. **학습 그래프**: `qdeploy.build`.
- 구조 추적 입력은 영상 대신 x1 = `torch.rand(1,3,256,256, generator seed 7919)`다. CPU fp32를 쓴다.
- **Z 텐서 집합**: `Graph.targets`(실수, dim ≥ 2)에서 두 종류를 뺀다. 채널 축은 `Graph.axis`다.
  - (a) `ad.is_train_only(n)`인 텐서(PBE)
  - (b) **실행되지 않은 텐서.** 규칙은 다음과 같다. n은 `ad.infer(model, x1).float().sum()`의 역전파에서 grad가 None이 아니면 실행된 것으로 본다. 또 그 소유자가 F3 forward hook이 실행을 본 leaf Conv/Linear이면 실행된 것으로 본다. 둘의 합집합이다(`used_param_names`).
  - 자동미분이 실패하면 hook 규칙만 쓰고 실패 수를 보고한다.
  - 제외 텐서 수와 이름은 모델별로 보고한다.
  - 민감도로, 제외 없는 전체 targets의 Z도 보고한다.
- **Mstat 층 집합**: qdeploy l50 층이다. `Graph(dep).leaf` 중 hook으로 실행이 확인되고 weight dim ≥ 2인 것이다.
- **무결성 대조.** F3 기록의 구조 필드만 읽는다. 다음이 같아야 한다.
  - `n_folded`, `params_inf_dep/train`
  - S_w4g·O_w4의 `n_tensors`·`max_levels`
  - D_w4의 `n_tensors`·`n_tensors_axis_size1`·`n_axis1_tensors`·`max_abs_level`
  - D_l50 `n_layers`(정확히)와 `min_zero_frac`(|차이| ≤ 1e-4)
  - 하나라도 어긋나면 `STOPPED_INTEGRITY`를 쓰고 멈춘다.

### 3.2 정의 (`theoryq_core.py`, `theoryq_run.py`)
- z_l = |{c : W_{l,c} ≢ 0이고 Q(W_l)_{l,c} ≡ 0}|/C_l이다. 판정은 연산자 출력에서 직접 한다.
- **Z(4) = max_l z_l** (주; 배포 그래프, w4_pt). Z_O(4)는 학습 그래프에서 같은 값이다. **Z_pc(4)**는 배포 그래프, w4_pc에서 같은 값이다.
- **무해 0과 Z′ (보조).**
  - 허용오차를 먼저 정한다. BN 채널은 **|γ_c| ≤ 1e-3·max_{c′}|γ_{c′}|**(같은 BN 모듈 안)일 때 "꺼짐"이다.
  - 학습 그래프 사본에서 꺼진 채널의 γ_c를 정확히 0으로 바꾸고, 같은 F3 코드로 접거나 융합한다.
  - 그 사본에서 가중치 채널이 정확히 0이 되는 배포 채널(원본에서는 0이 아님)이 **무해 채널 H_l**이다. 이 방식은 대응 관계를 따로 정할 필요가 없고, RepConv에서는 세 가지 BN이 모두 꺼져야 해당된다.
  - z′_l = |영점 채널 \ H_l|/C_l이고, **Z′ = max_l z′_l**이다.
  - 접지 않는 GN·LN·BN은 대상이 아니다.
- **Zw (보조)** = max_l(영점 채널의 ‖W_{l,c}‖² 합/‖W_l‖²_F).
- 그 밖의 보조 통계: Zglob(전 층 영점 채널 합/채널 합), Zelem(원소 단위 영점 비율의 max).
- **m_l = ‖W_l − prune_layer(W_l, 0.5)‖²_F/‖W_l‖²_F**, **Mstat = max_l m_l** (주). 보조는 E = Σ√m_l, Msum이다.
- 크기 기준선은 log10(params_inf_dep)이다.
- 진단값: 문턱 근처 채널 수(|r_c/A − 1/14| ≤ 1e-5), inf/NaN 텐서 수, 상위 3층.

---

## 4. 라벨과 셀 (F3 동결 함수, 변경 없음)
- 라벨: `inclusion → ckpt_rows → cell_table`을 쓴다. **붕괴 ⇔ 5-split 평균 val-Dice retention < 0.8**이다. malunet/isic18/k4는 제외되어 그 셀은 4 split이다.

| 가설 | 라벨 설정 | 셀 | 붕괴 | 비붕괴 |
|---|---|---|---|---|
| H_T1 | S_w4g | 66 | 38 | 28 |
| H_T2 | D_w4 | 66 (329 ckpt) | 9 | 57 |
| H_T3 | D_l50 | 66 | 55 | 11 |
| 보조 | O_w4 → S_w4g | 41 | 13 | 28 |

- **셀 통계 = 포함된 split의 산술 평균**이다.
  - 이유: 라벨이 같은 split들의 retention 평균이다. 또 max는 split 수에 따른 선택 편향이 있다.
  - max 집계는 민감도로만 쓴다.

---

## 5. 가설과 판정

### 5.1 규칙
- AUC: `qdeploy_analyze.auc`(동률 0.5, 큰 점수 = 붕괴)를 셀 단위로 쓴다.
- **셀 군집 부트스트랩**: 66셀을 복원추출한다. B = 2000, seed 20261017이고, 백분위 지수는 q_robust와 같다.
- **PASS ⇔ 95% CI 하한 ≥ 문턱**(H_T1 0.85, H_T3 0.75). 코디네이터 결정이다.
- **Holm**({H_T1, H_T3}, α = 0.05)은 함께 보고한다. p_h = (1 + #{AUC* ≤ 문턱_h})/(n + 1)이다(H0: AUC ≤ 문턱).
  - CI 하한 ≥ 문턱이면 대략 p ≤ 0.025이므로 두 규칙이 대개 같다.
  - 다르면 둘 다 적되, 판정은 CI 규칙을 따른다.
- p(H0: AUC ≤ 0.5)도 보고한다.

### 5.2 가설
- **H_T1**: Z(4) 대 S_w4g 셀 라벨, CI 하한 ≥ 0.85. 예측 0.6.
- **H_T3**: Mstat 대 D_l50 셀 라벨, CI 하한 ≥ 0.75. 예측 0.5. 비붕괴가 11셀뿐이라 CI가 넓을 것이다.
- **H_T2**: 포함된 329 checkpoint 전부에서 Z_pc(4) = 0이고 inf/NaN 텐서가 0개다(Prop 2의 확인).
  - D_w4 붕괴 9/66셀은 Z로 설명되지 않는다. cmunext cvc·etis, egeunet etis, lbunet cvc·etis, mhorunet etis, pbe etis, tinyunet etis, utanet etis이고, 모두 S_w4g에서도 붕괴한다.
  - 원고에는 "채널별 눈금으로도 남는 붕괴는 숨은 가지치기가 아닌 기전"인 **한계**로 쓴다.
  - Z_pc > 0이면 코드나 명제가 틀린 것이다. 다른 결과를 원고에 넣기 전에 원인을 찾는다.

### 5.3 보조 (검정 없음)
- **학습 → 배포 그래프 Z 변화.** ΔZ_cell = mean_k[Z_dep − Z_O]로 둔다.
  - (a) O_w4 비붕괴 41셀에서 새 붕괴 13셀에 대한 ΔZ의 AUC
  - (b) 접기·융합이 있는 7모델(pbe, maresunet, utanet, mk_unet, tinyunet, cmunext, lbunet)의 25셀로 좁힌 같은 AUC. 접기가 없는 4모델은 ΔZ ≡ 0이다.
  - (c) ΔZ > 0인 셀 수, 그중 새 붕괴 수
- Z′·Zw·Zglob·Zelem·전체 targets Z·log10 params 대 S_w4g, 그리고 Z_O 대 O_w4, E·Msum·log10 params 대 D_l50의 셀 AUC와 CI를 보고한다.
- max 집계 AUC를 보고한다.
- checkpoint 단위 AUC(q_robust `cluster_boot`와 같은 알고리즘)를 보고한다.
- 모델 내 AUC를 보고한다.
- 모델별 제외 텐서 수, 무해 채널 수, 시간을 보고한다.
- F3 agreement AUC(checkpoint 단위, D_w4 .935, D_l50 .982)와는 단위가 다르다는 각주를 붙여 비교한다.

### 5.4 실패 분기 (설계 §2)
- H_T1 FAIL: Prop 1은 부록으로 옮긴다. 예측기는 음의 결과로 보고한다. R-Q1은 Nagel 2019 인용으로만 쓴다.
- H_T3 FAIL: Prop 3은 상한으로만 남기고, R-Q2는 §8로만 판단한다.
- H_T2 FAIL: 원인 확인 전에는 원고에 넣지 않는다.

---

## 6. 합성 역검사 (`theoryq_synth.py`)

동결 전 로컬 CPU 실행에서 13/13 PASS였다. **동결 뒤 서버에서 동결 코드로 한 번 더 실행한다**(`run_theoryq.sh synth`).

| 검사 | 내용 |
|---|---|
| S1 | 심은 작은 범위 채널(r/A = 1/20, 1/30, 1/50. 1/13은 문턱 위), 원래 0인 채널 제외, conv와 ConvTranspose 축 → 3/16 |
| S2 | 문턱 A/14(코드) 대 A/15(설계). 차이 구간을 실증 |
| S3 | 채널별 RTN. 범위비 1e6에서도 영점 채널 0 |
| S4 | 해석식 {0 < r ≤ A/14} = 코드, 무작위 300 텐서에서 불일치 0 |
| S5 | Z = max_l, Zglob |
| S6–S7 | m_l 해석값(오차 < 1e-6), 순위, Spearman 1 |
| S8 | AUC 완전 1.0 / 역 0.0 / 무작위 ≈ 0.5 / 동률 0.5, 부트스트랩 CI와 p |
| S9–S10 | 4-split 셀 평균, Holm 손계산 |
| S11 | Prop 3 상한 성립(toy DAG, 50/50), 느슨함 1.1e-5 |
| S12 | 무해 채널: γ = 1e-4, 2.9e-3 → 무해. 3.1e-3 → 아님. γ = 0 채널은 원래 0이므로 세지 않음. 원본 불변 |
| S13 | 실행 규칙: 쓰이지 않은 가지·파라미터 제외, hook 합집합, grad 정리 |

**수정 기록.** 첫 실행에서 FAIL이 있었다. 모두 검사 쪽 구성 오류였다.
- S1 기대값 4 → 3
- S7 구성 두 번 교체
- S12 경계값 3e-3이 fp32에서 문턱과 같았다. 2.9e-3과 3.1e-3으로 바꾸고, float 비교를 고쳤다.

예측기 정의는 이 수정들로 바뀌지 않았다.

---

## 7. 실행

### 7.1 장소와 자원 (코디네이터 결정)
- 실행 장소: gpu-server, `<WORK>/theoryq/`. 쓰기는 이 폴더에만 한다. checkpoint·F3 기록·코드는 읽기만 한다.
- 자원: **CPU 전용**(`CUDA_VISIBLE_DEVICES=""`, 스크립트가 assert), 2 threads, `nice -n 10`.
- GPU 0·1(SGR·F4)은 건드리지 않는다.

### 7.2 파일과 1회 규칙
- 파일: `theoryq_core.py`, `theoryq_run.py`, `theoryq_decide.py`, `theoryq_synth.py`, `run_theoryq.sh`, `frozen_src/{qfix_a,qdeploy,qdeploy_analyze}.py`(서버 원본 사본, sha 동일).
- run이 시작할 때 F3의 `f2_fuse.py`, `f2b_fold_all.py`, `FOLD_PAIRS.json`, `MANIFEST_330.sha256`, `ITEMS.json` sha를 검사한다. checkpoint마다 manifest sha도 확인한다.
- `FROZEN_SHA256SUMS`에는 이 문서의 서버 사본(`PREREG_THEORYQ_FROZEN.md`), 위 스크립트 전부, 모델 코드 `qfix/A/code/models/**/*.py`, F3 fusion 파일의 sha를 적는다. `run_theoryq.sh`가 매 모드마다 `sha256sum -c`로 검사한다.
- 순서: synth 1회 → run(330, 재시작 안전, cap 3 h) → `ALL_DONE` → **decide 1회**.
  - decide는 RESULT 또는 `DECIDE_STARTED`가 있으면 거부한다.
- 중간 보고는 없다.

### 7.3 구조 dry run (동결 전, 값 비출력)
- 11 checkpoint 전부에서 F3 구조 필드와의 불일치가 0이었다. 자동미분은 22/22 성공했다.
- 제외 텐서(etis k1):
  - PBE: train_only 7(배포)
  - UTANet: 실행되지 않은 텐서 2(학습·배포 각 2)
  - 나머지 9모델: 0
- 시간은 checkpoint당 0.1–2.7 s(구성과 대조만)였다. 예측기를 포함한 330 checkpoint 전체는 30분 미만으로 추정한다.

---

## 8. SGR-E 후속 — **조건부. 별도 동결이 필요하며 이 문서로는 실행하지 않는다**
- **무엇.** SGR 단계 2(귀속 기반 재배분)를 **Prop 3 목적 Σ_l √m_l의 최소화**로 바꾼다. 단계 1·3·4와 영상·손실·스텝은 SGR과 같다.
- **문제.** k_l ∈ [0, ⌊0.9 n_l⌋]에 대해 Σ_l k_l = ⌊0.5 N⌋을 지키며 Σ_l √m_l(k_l)을 최소화한다.
  - m_l(k)는 |w| 오름차순(동률은 평탄 index)으로 k개를 0으로 했을 때의 상대 에너지다.
- **해법 (고정): 라그랑지안 + 잔차 탐욕.**
  - λ마다 층별로 k_l(λ) = argmin_k[√m_l(k) − λk]를 구한다(동률은 가장 작은 k). 누적합으로 벡터 계산한다.
  - λ를 이분법(60회)으로 정해 Σk_l(λ) ≤ K인 최대값을 찾는다.
  - 잔차 K − Σk_l은 한계비용 √m_l(k + 1) − √m_l(k)가 가장 작은 층부터 하나씩 채운다. 동률은 층 실행 순서로 깬다. 이렇게 하면 Σk_l = K가 정확히 성립한다.
  - √m_l(k)는 k에 대해 볼록이 아니다(볼록 누적합의 제곱근). 라그랑지안은 하위 볼록껍질 위의 해를 찾을 뿐이므로 **전역 최적은 보장하지 않는다.**
  - 이 목적은 층을 상한 0.9까지 몰아 가지치는 경향이 있을 수 있다. 층별 희소도 분포를 보고한다.
- **LAMP와의 관계.** Σ m_l 목적은 층 정규화 크기 점수(LAMP, Lee 외 ICLR 2021과 가까움)의 정확한 해가 된다. **비교 arm으로 돌리지 않는다**(코디네이터 결정).
- **큐와 cap.** SGR `ALL_DONE`과 decide 뒤의 별도 큐다. D_l50 붕괴 55셀(275 checkpoint)을 SGR 순서로 돌리고, cap은 **+4 GPU-h**다.
  - SGR이 저장한 L\*·g_l이 있으면 재사용한다.
  - cap에 걸리면 완료된 셀에서만 SGR과 짝지어 비교하고 `partial`로 표시한다.
- **채택 규칙.** SGR-E 복구율 ≥ SGR 복구율 − 5%p이면 SGR-E를 채택한다(더 단순한 쪽). 복구 정의는 SGR과 같다.
- **동결 전 단위 검사(CPU).** 전역 수가 정확한지, 상한을 지키는지, 결정론적인지 본다. 또 Σ√m(SGR-E) ≤ Σ√m(record l50)인지와 Σ√m(SGR-E) ≤ Σ√m(SGR 마스크)인지 확인한다.

---

## 9. 결정된 선택 (코디네이터 2026-10-17)
1. PASS = CI 하한 ≥ 문턱. Holm은 보고한다.
2. 셀 통계 = 평균. max는 민감도다.
3. 실행되지 않은 텐서는 제외한다(§3.1 규칙). 무해 0을 뺀 Z′는 보조다(γ 허용오차 1e-3 상대).
4. Zw는 보조다.
5. 서버 CPU에서 실행한다.
6. SGR-E는 Σ√m_l, 라그랑지안 + 잔차 탐욕이다. Σm_l(≈ LAMP)은 돌리지 않는다.
7. Prop 3은 실제 모델에 대해 순위 근거일 뿐이다.
