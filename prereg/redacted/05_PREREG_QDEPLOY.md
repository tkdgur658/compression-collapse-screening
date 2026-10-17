# PREREG_QDEPLOY — F3: 배포 그래프(RepConv fused, BN folded)에서 Q 재측정

**FROZEN 2026-10-16 15:39 KST (sha256은 `prereg_release/SHA256SUMS.txt`와 서버 `qdeploy/FROZEN_SHA256SUMS`).** 코디네이터 결정(18:3x) 5항을 §F에 반영했다. 동결 시점까지 측정 실행(`--run`) 0회. 동결 전에 한 일은 manifest 생성, fold pair 탐색(무작위 BN stats), 합성 역검사뿐이며 Dice·R·AUC는 출력하지 않았다(AMENDMENT 0). 이후 수정은 말미 AMENDMENT로만 한다. 본문과 §F가 다르면 §F가 우선한다.

근거 문서: `journal_review_1/SUMMARY.md` §2 #1·§3 F3·§6, REVIEWER_1 M3·minor 11·E3, REVIEWER_2 M1·E1·E8, REVIEWER_3 M3·M4, `iso/s4/mem/AUDIT/PREREG_35.md`·`RESULT_35.md`·`p3_agree.py`, `iso/s4/qfix/A_PLAN.md`·`B_fix_design.md` §1·`C_PLAN.md`, 서버 `<WORK>/qfix/C/rec/qfix_a.py`(sha256 b36df82f…94c6)·`eval/eval_C1.py`(152694ec…1166), `manuscript_cleanup/F2_efficiency/f2_fuse.py`·`f2_fuse_out.json`·`f2_fuse_fp64_out.json`.

---

## 0. 먼저 적어 두는 사실 (설계에 영향)

1. **"PBE-Net 11/18 붕괴"의 출처가 이 사전등록의 라벨과 다르다.** 원고의 11/18은 P3용 `queue10_pbe_reference_k1_v1` checkpoint(v1 runs와 sha 불일치), split 1, chance+0.05 규칙에서 나왔다(`B_fix_design.md` §1 "정정 범위"). 같은 v1 30 runs에 retention < 0.8(5-split 평균) 라벨을 쓰면 원 설정의 PBE-Net은 **18/18 붕괴(0/18 OK)** 이다(`B_baseline_18cells.json`, sha eaadb4cc…5250). 따라서 결정 (i)의 "원 설정" 열은 **이 실행에서 같은 330 checkpoint·같은 라벨로 다시 계산한 값**을 쓰고, 원고의 11/18은 정의를 붙여 각주로만 병기한다(§6).
2. **fp32 절대 1e-4 등가성은 정상 fusion에서도 깨진다.** F2의 PBE-Net fusion(`f2_fuse.py`)은 fp32에서 max|Δlogit| = 1.68e-4(isic18, |logit| 최대 180.8)이고 fp64에서는 3.4e-13이다. 그래서 decisive gate는 fp64에서 둔다(§3.4, 미해결 선택 Q1).
3. **베이스라인 10개 중 자체 re-parameterization/deploy 코드가 있는 모델은 없다**(`code/pbe_seg/models/*` grep: `switch_to_deploy`·`reparam`·`fuse` 0건). GroupNorm·LayerNorm을 쓰는 모델은 egeunet, lbunet, malunet, mhorunet, ucmnet, mk_unet이고, BN 접기 대상은 BatchNorm2d뿐이다.
4. 서버 checkpoint 현황(2026-10-16 listing): 베이스라인 10모델 × 6 ds × k1–5 = **300개 checkpoint.pth 모두 존재**. 이 중 6개 run 폴더에는 오래된 `FAILED.txt`가 남아 있지만 `results.json`은 `status: complete`, protocol v1이다(egeunet kvasir k2·k3, malunet isic18 k4, mhorunet isic18 k3, tinyunet kvasir k4, utanet isic17 k3). 베이스라인 코드 commit은 e0e2f07 196개, fbe35ca 104개. PBE v1 = archive 20 + `pbe_v1_fill` 10 = 30(commit e0e2f07).

## 1. 가설 (사전 기대 방향 포함)

**H_D:** 배포 그래프(RepConv fused, BatchNorm folded)에 per-channel w4, a6, l50을 적용해도 11 모델 × 6 데이터셋 × 5 split에서 붕괴(retention < 0.8)가 남는다.
- 판정 대상의 중심은 PBE-Net 18셀(6 ds × 3 setting)이다. 베이스라인 180셀은 같은 규칙으로 함께 보고한다.
- 사전 기대(판정에 쓰지 않음): 원고 부록 W4-ch(per-channel 4-bit에서 PBE-Net이 6 ds 중 5개에서 IoU 0.08 이내)와 Z1(미융합 1×1 depthwise 분기 zeroed fraction 0.18–0.19)로 보아 w4_pc 붕괴는 크게 줄 것으로 본다. a6·l50은 그래프 변경만으로 줄어든다는 근거가 약하다. 예측: PBE-Net D 붕괴 셀 수 중앙값 7/18, 80% 구간 3–12.

## 2. 자산과 해시 검증

### 2.1 checkpoint 목록 (330)
| 집합 | 경로 (gpu-server) | 수 | 포함 규칙 |
|---|---|---|---|
| 베이스라인 | `<ARCHIVE>/runs/main/<model>/<ds>/k<k>/checkpoint.pth`, model ∈ {maresunet, utanet, mk_unet, mhorunet, tinyunet, ucmnet, lbunet, cmunext, egeunet, malunet}, ds ∈ {PH2, isic17, isic18, kvasir, cvc, etis}, k ∈ 1..5 | 300 | `results.json`의 status == complete, `config.protocol` == v1 PROTOCOL, `config.model` == registry 이름, `model_cfg` == adapter DEFAULT_CFG. 오래된 `FAILED.txt`가 있는 6개 run은 **§F.5 조건(sha 일치 + val IoU 재현 1e-3)을 만족할 때만 포함**하고, 아니면 제외·기록한다 |
| PBE-Net v1 | `<ARCHIVE>/runs/main/pbe/<ds>/k<k>` (20, `isic17/k2`는 FAILED·results.json 없음 → 제외) + `<WORK>/pbe_v1_fill/runs/pbe/<ds>/k<k>` (10) | 30 | `qfix_a.py:90-107` `list_runs()`와 같은 규칙(assert 30) |

영상 캐시: `<ARCHIVE>/cache/v1/<ds>/{val,test}_<k>_<key>.npz`, 파일명은 각 run의 `results.json` `data_cache`에서 읽는다(archive는 읽기만; 캐시가 없으면 run을 FAIL로 표시하고 생성하지 않음).

### 2.2 해시 검증 절차 (측정 전, CPU, read-only)
1. **Manifest 생성**: 330개 `checkpoint.pth`와 `results.json`의 sha256을 계산해 `BASE/MANIFEST_330.sha256`에 쓴다(`BASE=<WORK>/qdeploy`). 이 파일의 sha256을 동결본에 적는다.
2. **기존 기록과 대조**: (a) PBE 30 = stage A 기록 `qfix/A/out/<ds>_k<k>.json`의 `ckpt_sha256` 필드와 30/30 일치해야 한다. (b) 베이스라인 k1 60 = P3 기록(`mem/AUDIT/P3_raw.json` 무결성, 9901 기록 해시와 66/66 일치)과 대조한다. P3는 `fieldswitch/ckpt/<model>/<ds>/k1` 복사본을 썼으므로 archive k1과 sha가 같아야 한다. (c) k2–k5 베이스라인 240개는 이전 해시 기록이 없으므로 manifest 동결 시점 해시가 기준이 된다.
3. **실행 중**: Engine은 load 시 sha256을 다시 계산한다(`qfix_a.py:174-178` 방식). manifest와 다르면 해당 checkpoint를 FAIL로 표시하고 결과에서 뺀 뒤 수를 보고한다. PBE checkpoint가 하나라도 다르면 실행을 중단한다.
4. **코드 일치**: 베이스라인 model 소스 디렉터리가 e0e2f07(`pbe_seg/code`)과 fbe35ca(`pbe_seg/code_v2`) 사이에 byte-identical인지 확인한다. 300개 모두 e0e2f07 registry로 `strict=True` load가 되어야 한다. 하나라도 실패하면 측정 전에 보고하고 멈춘다.

## 3. 배포 그래프 구성 (training graph → deployment graph)

### 3.1 공통 원칙
- 순서: **fuse → fold → 등가성 검증 → 양자화/가지치기.** 양자화와 가지치기는 접힌 가중치(BN scale이 곱해진 가중치)에 적용한다. 이것이 배포 툴체인의 순서다.
- 접힌 bias와 남은 normalization 연산은 fp32로 둔다(int32 bias 관례에 해당). dim < 2 텐서는 양자화 대상이 아니다(record 규칙과 같음).
- train-only 모듈(PBE `shead1-3`, `base.gt_conv1-4`; 각 adapter `is_train_only`)은 추론 그래프에서 실행되지 않으므로 다루지 않는다.

### 3.2 PBE-Net (명시 규칙, `f2_fuse.py`와 같은 연산)
| 대상 (net.py, 서버 e0e2f07 기준) | 규칙 |
|---|---|
| `RepConv` 9개(encoder1–5, decoder1–4의 `[0][*]`, net.py:22-38) | `conv_kxk+bn_kxk`, `conv_1x1+bn_1x1`(3×3 중앙으로 zero-pad), `bn_identity`(중앙 1, groups=C)를 각각 접은 뒤 kernel과 bias를 더해 3×3 depthwise conv 1개(bias 있음) + ReLU로 바꾼다(`f2_fuse.py` `fuse_rep`) |
| `EBPB.branch3`(net.py:62-64) | conv–BN 2쌍을 접는다(`fuse_conv_bn`) |
| `DWSepBottleneck`(net.py:88-95) | `dw+bn1`, `pw+bn2`를 접는다 |
| 1×1 stage 입력 conv(`stage()` net.py:113-119의 첫 conv) | BN이 뒤따르지 않으므로 그대로 둔다 |
| `PRCM`(net.py:41-52) | `basis`(8×C)와 `fuser`(C×8) 두 행렬곱을 그대로 둔다(C×C 하나로 합치지 않는다. 합치면 파라미터가 16C에서 C²로 늘어나 배포 관례가 아니다). eval에서 dropout = identity |
기대 결과: BatchNorm 33 → 0(`f2_fuse_out.json` n_bn_fused 0), 추론 파라미터 1,334,337 → 1,318,081.

### 3.3 베이스라인 10개 (일반 규칙; 자체 deploy 코드 없음)
- **접기 대상**: eval 모드 `BatchNorm2d`가 `Conv2d` 또는 `ConvTranspose2d`의 출력을 **바로** 입력으로 받는 쌍. ConvTranspose2d는 출력 채널 축(weight dim 1, groups=1)으로 scale을 곱한다. Conv bias가 있으면 접기에 포함한다.
- **쌍 찾기**(모델당 1회, 측정 전): 각 모델의 etis k1 checkpoint 사본에서 BN running stats를 무작위화(seed 7919)한 뒤 forward hook으로 BN 입력 텐서와 같은 storage·shape·stride를 가진 conv 출력을 후보로 잡는다. 후보 쌍마다 하나씩 접어 fp64 출력 차가 1e-9·max|logit| 이하일 때만 채택한다. 같은 conv 출력을 다른 가지가 함께 쓰는 경우는 이 검사에서 걸러진다. 채택 목록은 `BASE/FOLD_PAIRS.json`에 쓰고 그 sha256을 동결 AMENDMENT 0에 적는다(측정 전).
- **접지 않는 것**: GroupNorm, LayerNorm(데이터 의존), conv 앞에 오는 BN(pre-activation), 위 검사에서 탈락한 BN. 이들은 fp32로 남기고 모델별 수를 보고한다.
- 구조적 re-parameterization 분기가 있는 베이스라인은 없으므로 branch fusion은 PBE-Net에만 적용된다(미해결 선택 Q3).

### 3.4 등가성 검증 (checkpoint마다, 양자화 전)
- 입력: 그 run의 test 캐시 앞 8장 + val 캐시 앞 8장(저장 순서).
- **E1 (decisive)**: training graph와 deployment graph를 둘 다 float64로 바꿔 계산한 logit의 max|Δ| ≤ 1e-4. 실패한 checkpoint는 D·S 설정에서 빼고 수를 보고한다. PBE-Net이 하나라도 실패하면 실행을 중단하고 보고한다.
- **E2 (보고만)**: fp32 max|Δ|, 상대 max|Δ|/max|logit|, 마스크(sigmoid > 0.5) 불일치 픽셀 비율. 사전 기대: 상대 ≤ 1e-5, 불일치 ≤ 1e-4.
- fp32 val Dice는 두 그래프에서 모두 계산하고 차이를 보고한다. D·S의 retention 분모는 deployment graph의 fp32 Dice다.

## 4. 연산자 (O = 원 설정 재계산, D = 결정 설정, S = 민감도)

record 연산자는 `qfix_a.py:110-141`(`rtn_sym_tensor`, `act_hook_factory`, `prune_layer`)를 그대로 import한다. 새 연산자는 같은 파일 안 형식(같은 rounding, clamp, 0 보호)으로 쓴다.

| 이름 | 그래프 | 정의 | 역할 |
|---|---|---|---|
| **O-w4** = w4_pt | training | per-tensor symmetric RTN 4 bit, scale = max\|w\|/7, levels −7..7, 대상 = dim ≥ 2인 모든 float parameter, BN 미접힘 | 원 설정 |
| **O-a6** = a6 | training | 모든 Conv/ConvT/Linear 출력(pre-BN)에 dynamic per-tensor symmetric 6 bit fake-quant, amax는 샘플별 | 원 설정 |
| **O-l50** = l50 | training | 실행된 Conv/Linear weight(dim ≥ 2)에 layer-wise unstructured magnitude pruning 50 %, k = floor(0.5n), \|w\| ≤ k번째 값 → 0 | 원 설정 |
| **D-w4** = w4_pc | deployment | **per-output-channel symmetric RTN 4 bit**: 출력 채널 c마다 scale_c = max\|w_c\|/7(Conv2d·Linear은 dim 0, ConvTranspose2d는 dim 1, 0이면 1). 대상 = deployment graph의 dim ≥ 2 float parameter 전부(PRCM `basis`는 행 = 출력 축 dim 0). 모듈 밖 Parameter(예: (1,C,H,W) 형태)는 dim 0 기준이며, dim 0 크기가 1이면 결과적으로 per-tensor가 된다. 이런 텐서 수를 보고한다 | **결정** |
| **D-a6** = a6_pt | deployment | O-a6과 같은 수식(dynamic per-tensor 6 bit)을 deployment graph의 모든 Conv/ConvT/Linear 출력에 적용한다. 접힌 conv의 출력은 BN 이후 값이다 | **결정** |
| **D-l50** = l50_lw | deployment | O-l50과 같은 수식(layer-wise 50 %)을 deployment graph의 실행된 Conv/ConvT/Linear weight에 적용한다(접힌 가중치, fused 3×3 kernel 포함) | **결정** |
| S-w4g | deployment | per-tensor w4(그래프 효과만) | 민감도 |
| S-w4c | training | per-channel w4, BN 미접힘(granularity 효과만, 부록 W4-ch에 해당) | 민감도 |
| S-a6c | deployment | dynamic **per-channel** 6 bit: (샘플, 출력 채널)마다 amax | 민감도 |
| S-l50c | deployment | per-output-channel 50 % magnitude pruning: 출력 채널마다 k_c = floor(0.5·n_c) | 민감도 |

- D의 a6를 per-tensor로 둔 이유: 정수 커널 대부분은 활성값 per-tensor scale만 지원하고, 이렇게 하면 a6에서는 그래프 하나만 바뀐다. per-channel은 S-a6c로 보고한다(미해결 선택 Q2).
- D의 l50을 layer-wise로 둔 이유: record와 같은 수식이어서 그래프 효과만 분리된다. per-channel은 S-l50c로 보고한다(미해결 선택 Q2b).
- 입력 영상은 양자화하지 않는다(record와 같음). 마스크 = sigmoid(logit) > 0.5. 입력은 256×256, uint8/255. FP32, TF32 off, cudnn deterministic.
- 무결성 카운터(checkpoint·설정마다): 대상 텐서 수, 채널별 고유 레벨 수 최댓값(≤ 15), hook 수, 가지치기 층 수와 최소 zero 비율(≥ 0.5 − 1/n), 가중치 byte-exact 복원.

## 5. 지표

- **Dice per image** = 2TP/(2TP+FP+FN). 둘 다 비면 1(record IoU u = 0 → 1 규칙). val Dice = run의 val split 전체 평균. IoU도 함께 기록한다.
- **retention** R = Dice_q / Dice_fp32 (val). O의 분모는 training graph fp32, D·S의 분모는 deployment graph fp32다.
- **셀** = (model, ds, setting). 셀 값 = 5 split 평균 R(`B_fix_design.md` §1과 같은 정의). **붕괴 라벨 2개**: L80 ⇔ 셀 평균 R < 0.8 (결정용), L90 ⇔ < 0.9 (보고). checkpoint 단위 수(330 × 3)도 함께 보고한다.
- **라벨 없는 출력 일치도(P3 방식)**: 그 run의 test 캐시 `images` 앞 8장(저장 순서, `masks` 키 미접근). agree = mean_i Dice(fp32 mask_i, q mask_i), 둘 다 비면 1, 한쪽만 비면 0, NaN/Inf logit은 False(`p3_agree.py`, `qfix_a.py:247-251`). D 일치도의 fp32 기준은 deployment graph다. fp32 마스크가 빈 test 영상 수를 보고한다(F1).
- **대조 점수 1, 전경비율**: f = 8장 평균 예측 전경 비율. c_fg = 1 − min(f_q, f_fp32)/max(f_q, f_fp32), 둘 다 0이면 0. 공간 일치 없이 "마스크가 비거나 가득 찼는가"만 본다.
- **대조 점수 2, 출력 SQNR**: SQNR_dB = 10·log10(Σ logit_fp32² / Σ (logit_fp32 − logit_q)²)을 8장 전체에서 float64로 계산한다. 오차가 0이면 100 dB, 비유한 logit이면 −100 dB로 둔다.
- **AUC**: Mann–Whitney, 동점 0.5. 붕괴일수록 큰 점수를 쓴다(1 − agree, c_fg, −SQNR). **split별 계산**: split k에서 66 checkpoint × 3 setting = 198 cell-checkpoint를 쓴다. 라벨은 checkpoint 단위 R < 0.8(그리고 < 0.9)이다. **보고 = 5 split 평균 AUC와 split별 [min, max]**. 양성이나 음성이 0인 split은 제외하고 그 수를 적는다. 설정별 AUC는 양성이 5개 미만이면 "—(n)"으로 적는다(R2 #18). 보조: 990개 풀링 AUC와 checkpoint 단위 cluster bootstrap 95 % CI(2000회, seed 7919), Spearman ρ(점수, R)의 설정별 값.
- 위 지표 전부를 O, D 두 묶음에서 따로 계산한다(S는 셀 수와 R만 보고).

## 6. 판정 규칙 (코디네이터 결정 그대로; 결과 전 고정)

**(i) 나란히 보고.** 아래 표를 L80·L90 각각에 대해 낸다. 이 실행에서 O를 다시 계산한 값과 D를 같은 checkpoint·같은 라벨로 나란히 둔다. 원고의 "11/18"은 각주에 출처와 함께 적는다("queue10 k1 checkpoint, chance+0.05 규칙").

| 묶음 | 전체 붕괴 셀 /198 | PBE-Net /18 (w4·a6·l50) | 베이스라인 /180 | checkpoint 단위 /990 |
|---|---|---|---|---|
| O (원 설정, 재계산) | | | | |
| D (배포 그래프) | | | | |
| S-w4g, S-w4c, S-a6c, S-l50c (각 /66, PBE /6) | | | | |

**(ii) 분기(L80, D 기준, PBE-Net 18셀).**
- **N_PBE(D) ≤ 3/18** → 원고의 PBE-Net 붕괴 사실을 "원 설정(미융합·BN 미접힘·per-tensor)의 산물"로 다시 진술한다. C1–C3 절은 부록으로 옮긴다. Q 사실 자체(일치도가 붕괴를 식별함)의 존속은 (iii)의 수치로 따로 진술한다.
- **N_PBE(D) ≥ 4/18** → PBE-Net 취약성이 배포 그래프에서도 남는다고 진술한다. 본문 Q 표는 D를 주 설정으로, O를 스트레스 설정으로 바꾼다. C1–C3의 위치는 §6 기존 결정(C3 = Results의 사후 파일럿)을 따른다.
- E1 탈락으로 PBE 셀이 5 split을 채우지 못하면 판정은 PARTIAL이다. 부분 수를 그대로 보고하고 설계는 바꾸지 않는다.

**(iii) 함께 보고(결정 아님).** L80·L90 두 라벨의 셀 수, 그리고 agree, c_fg, SQNR의 AUC(5-split 평균 [range])를 O·D 각각에 대해 낸다. 사전에 정한 읽기 규칙(제안, Q5): D·L80에서 AUC_agree − AUC_fg ≤ 0.02이면 본문은 "일치도 점수는 전경비율 점검과 구별되지 않는다"고 쓴다. 차이가 > 0.02이고 5 split 모두에서 양수이면 "일치도가 전경비율 대조군보다 높다"고 쓴다. 그 사이면 수치만 보고한다. SQNR도 같은 규칙으로 비교한다.

## 7. 합성 역검사 (실데이터 점수 보기 전, 같은 함수)
1. **fusion 검사기**: 일부러 틀린 PBE fusion(`bn_identity` 분기 누락)과 틀린 BN 접기(running_var 대신 running_var² 사용)를 넣으면 E1이 FAIL해야 한다. 올바른 fusion은 E1 PASS(fp64 Δ < 1e-10)여야 한다. 합성 Conv+BN(무작위 stats)의 fp64 접기 오차 < 1e-12.
2. **양자화기**: 채널 0만 ×1000인 합성 weight(8×16×3×3)에서 per-tensor w4는 채널 1–7의 90 % 이상을 0으로 만들어야 한다. per-channel w4는 채널마다 상대 max 오차 ≤ 1/14, 채널별 고유 레벨 ≤ 15여야 한다. ConvTranspose2d 축(dim 1) 사례도 같은 조건을 만족해야 한다. a6 per-tensor/per-channel, l50 layer/channel의 zero 비율이 정의값과 같아야 한다.
3. **분석 함수**: 합성 R 표에서 PBE 붕괴를 2/18로 심으면 분기 (ii)가 "artifact", 5/18로 심으면 "survives"가 나와야 한다. score = 1−R³이면 AUC 1, score = R이면 0, 무작위면 0.5 ± 0.1이어야 한다. split별 평균·range 계산을 손 계산 사례로 맞춘다. 양성이 0인 split은 제외 처리되어야 한다.
4. **함수 보존(planted)**: PBE etis k1 사본에서 `base.encoder3.0.1.conv_kxk` 채널 0 ×1000과 그에 맞춘 BN 보정(A_PLAN §6과 같은 함수 보존 변환)을 넣는다. training graph fp32 Dice가 1e-3 이내로 같고, E1 PASS이며, 이 사본에서 D-w4(per-channel)의 R이 원본 대비 0.02 이내로 같아야 한다. per-tensor S-w4g의 R은 낮아져야 한다. 이 검사는 per-channel이 채널 이상치에 강하다는 성질을 확인한다.
출력은 `BASE/synthetic.log`에 "SYNTHETIC" 접두사로 쓰고 동결 AMENDMENT 0에 요약 1줄을 붙인다. 하나라도 FAIL이면 측정하지 않고 보고한다.

## 8. sha 기록·결과 1회·봉인
- 동결 전에 다음 sha256을 기록한다: `qdeploy.py`(engine + 연산자 + fusion), `qdeploy_analyze.py`(판정), `run_qdeploy.sh`, `rec/qfix_a.py`(b36df82f…94c6 확인), `FOLD_PAIRS.json`, `MANIFEST_330.sha256`. 실행 후 다시 계산해 같아야 한다.
- smoke(1개 checkpoint)와 timing gate는 wall time과 무결성 카운터만 출력하고, Dice·R·AUC는 출력하지 않는다.
- checkpoint별 JSON은 값을 봉인한 채 쓴다(`eval_C1.py`의 "sealed until --decide" 방식). 330개가 모두 DONE이거나 FAIL로 확정된 뒤 `--decide`를 **1회** 실행한다. 재실행은 crash 재개만 허용한다(DONE은 건너뜀). 버그 수정은 AMENDMENT로 diff를 기록한 뒤에만 하며, 그 경우에도 결과를 본 뒤의 설계 변경은 하지 않는다.

## 9. GPU 예산과 런타임 추정 (gpu-server GPU 0, cap 6 GPU-h)
- 실행: `CUDA_VISIBLE_DEVICES=0`, `flock BASE/gpu0.lock`, 프로세스 1개. 다른 사용자 job이 있는지 launch 직전 `nvidia-smi`로 확인한다. 종료는 PID로만 한다.
- checkpoint당 pass 수 = fp32 2(training·deployment) + O 3 + D 3 + S 4 = 12회(val 전체 + test 8장). 여기에 fp64 등가성 검사(16장)가 더해진다.
- 기준 시간: `eval_C1.py`는 PBE checkpoint당 52 pass에 etis 5.1 s, PH2 8.9, cvc 12.8, kvasir 20.0, isic17 81.3, isic18 101.8 s가 걸렸다(eval.log). pass당 0.10–1.96 s.
- PBE 30개: 5 × 12/52 × Σ_ds = 5 × 0.231 × 229.9 s ≈ **265 s**.
- 베이스라인 300개: pass당 비용을 PBE의 1.5배(중심)에서 3배(상한)로 가정하면 4.0k–8.0k s.
- load·fusion·fp64 검사: 약 5 s × 330 ≈ 1.6k s.
- **추정 = 중심 1.6 GPU-h, 상한 2.7 GPU-h.** 코디네이터 수치(checkpoint당 9–100 s를 그대로 적용)로는 330 × 38 s ≈ 3.5 h, 최악 9.2 h다. 다만 이 수치는 52-pass sweep 기준이어서 12-pass인 이 실행에는 과대 추정이다.
- **Timing gate**: 11모델 × {etis k1, isic18 k1}에서 시간만 재고 투영한다. 투영이 5.0 GPU-h 이하일 때만 launch한다. 실행 중 wall guard는 6.0 h이며, 넘으면 중단하고 PARTIAL로 보고한다(남은 checkpoint 목록 포함).
- 순서: PBE 30 → 베이스라인(ds 비용이 낮은 순: etis, PH2, cvc, kvasir, isic17, isic18). 중단되더라도 PBE 18셀 판정은 남는다.

## F. 동결 결정 (코디네이터 18:3x, 본문보다 우선)

1. **등가성 gate(Q1)**: decisive = E1, training·deployment graph를 float64로 계산한 logit max|Δ| ≤ 1e-4. fp32는 max 절대차, **max 상대차(max|Δ|/max|logit|)**, 마스크 불일치 비율을 **보고만** 한다. fp64 forward 자체가 실패한 checkpoint는 E1 FAIL로 센다(D·S에서 제외, O는 유지).
2. **주 설정(Q2, Q2b, Q7)**: D-a6 = dynamic per-tensor 6 bit, D-l50 = layer-wise 50 %(원 설정과 같은 수식, 비교 가능). D-w4 = per-channel. w4의 그래프 × granularity **2×2 분해(S-w4g, S-w4c)는 보조**로 보고한다. S-a6c, S-l50c도 보조(민감도)로 남긴다. 판정은 D 3설정만 쓴다.
3. **베이스라인 배포 그래프(Q3)**: 베이스라인에는 **일반 BN 접기만** 적용한다. 자체 re-parameterization이 있는 베이스라인은 없고, GroupNorm·LayerNorm과 접히지 않는 BN은 **fp32로 남긴다**(결과에 모델별 수 보고). fold pair 탐색 결과(FOLD_PAIRS.json, AMENDMENT 0): maresunet 48/48, utanet 45/46, mk_unet 62/62, tinyunet 72/72, cmunext 10/43, lbunet 2/5(+GN 10), mhorunet GN 10, ucmnet GN 6·LN 25, egeunet GN 10, malunet GN 40 (접힘/BN 총수). PBE-Net은 §3.2(fuse_pbe, BN 33 → 0).
4. **주 비교(Q4)**: (i)의 비교 기준은 **같은 330 checkpoint에서 같은 라벨(5-split 평균 val Dice retention < 0.8)로 다시 계산한 원 설정 O**다. 원고의 11/18(queue10 k1 checkpoint, chance+0.05 규칙, split 1)은 **각주로만** 적는다.
5. **오래된 FAILED.txt가 남은 베이스라인 6 폴더(Q6)**: 다음 두 조건을 **모두** 만족할 때만 포함한다. (a) checkpoint sha256이 manifest와 같고, 독립 사본(`<workspace>/fieldswitch/ckpt/<m>/<ds>/k<k>/checkpoint.pth`)의 sha256과도 같으며, results.json sha256이 Table 1 사본(`<workspace>/data/main/.../results.json`)과 같다. (b) 이 실행에서 training graph fp32로 다시 계산한 val mean IoU(영상별 IoU 평균, union 0 → 1, 임계 0.5)가 저장된 `best_val_iou`와 **1e-3 이내**다. 하나라도 어기면 그 checkpoint는 모든 묶음에서 제외하고 이유와 함께 기록한다. 참조 sha는 `qdeploy.py` `OLD_FAILED`에 동결 전에 적었다. (b)의 차이는 330개 전부에 대해 보고만 한다(6개 외에는 판정에 쓰지 않음).
6. **판정 규칙 (i)–(iii)**: §6 그대로(읽기 규칙 Q5의 AUC 차 0.02 포함). 분석은 모든 item이 DONE 또는 FAIL로 확정되어 `out/ALL_DONE`이 생긴 뒤 `run_decide.sh`로 **1회** 실행한다. 스크립트는 ALL_DONE이 없거나, STOPPED 표식이 있거나, 결과 파일이나 `DECIDE_STARTED`가 이미 있으면 거부한다.
7. **§7.4 정정(동결 전, 결과 보기 전)**: 초안의 기대 "S-w4g의 R이 낮아진다"는 틀렸다. BN 보정으로 함수를 보존한 plant는 BN 접기에서 정확히 상쇄되므로 deployment graph에는 이상치가 남지 않는다. 동결 판정 조건은 다음과 같다. PBE kvasir k1 사본(stage A 합성과 같은 run)에서 training graph fp32 Dice 차 < 1e-3, E1 PASS, |ΔR(D-w4)| ≤ 0.02(접기가 plant를 흡수), |ΔR(S-w4c)| ≤ 0.02(per-channel이 채널 이상치에 강함), 그리고 plant 텐서에 per-tensor w4를 걸면 채널 1.. 의 90 % 이상이 0이 된다(원 설정 연산자가 plant를 보는지 확인).
8. **§3.3 정정**: pair 탐색은 `f2b_fold_all.find_pairs`(BN 입력이 Conv/ConvT/Linear 출력 텐서와 같은 객체이고 두 모듈이 각각 1회 호출, 한 층이 BN 2개를 먹이면 제외)로 후보를 잡는다. 그다음 etis k1 사본을 float64로 바꾸고 BN stats를 무작위화한다(seed 7919; mean N(0,0.1²), var U(0.5,2), γ U(0.8,1.2), β N(0,0.1²)). 후보를 하나씩 접어 val 2장에서 max|Δlogit| ≤ 1e-9·max|logit|일 때 채택한다. checkpoint마다 같은 trace를 다시 하고, 채택 쌍이 그 checkpoint의 후보에 있는지 assert한다.
9. **실행·중단 규칙**: gpu-server GPU 0만(`CUDA_VISIBLE_DEVICES=0`, flock `gpu0.lock`·`qdeploy.lock`, nohup, `< /dev/null`). item별 `out/items/<model>__<ds>__k<k>.{json,DONE,FAIL}`. compute 합계가 6 GPU-h에 닿으면 `out/STOPPED_CAP`. OOM이 나면 그 item을 batch 4로 1회 재시도하고, 두 번째 OOM에서 `out/STOPPED_OOM`. GPU 0에 다른 프로세스가 있으면 60 s 간격으로 기다리며, 대기 시간은 cap에 넣지 않는다. PBE sha 불일치나 PBE E1 FAIL이면 `out/STOPPED_PBE_*`. 실행 전 `FROZEN_SHA256SUMS`를 `sha256sum -c`로 확인한다. Timing gate(§9)는 같은 엔진으로 11모델 × {etis k1, isic18 k1} 22 item을 실제 출력 폴더에 봉인한 채 돌리고(값 비출력), 투영 ≤ 5.0 GPU-h일 때만 나머지를 launch한다.

## 10. 코디네이터가 확인할 미해결 선택 (§F에서 모두 결정됨, 기록용)
- **Q1 등가성 gate**: fp32 절대 1e-4는 정상 fusion에서도 깨진다(F2 1.68e-4). 제안: decisive gate는 fp64 max|Δ| ≤ 1e-4, fp32는 상대 ≤ 1e-5와 마스크 불일치를 보고만 한다. 대안: fp32 상대 기준을 gate로 둔다.
- **Q2 a6 정의**: D-a6 = dynamic per-tensor(제안)와 per-channel(S-a6c) 중 무엇을 결정 설정으로 둘지. 또 static calibration(예: 같은 8장 calibration)으로 바꿀지(R2 E1). 제안은 dynamic 유지다. static으로 바꾸면 calibration 영상과 측정 영상을 분리해야 한다.
- **Q2b l50 정의**: D-l50 = layer-wise(제안, record와 같은 수식)와 per-channel(S-l50c) 중 무엇으로 할지.
- **Q3 베이스라인 배포 fusion**: 자체 deploy 코드가 있는 베이스라인이 없다. 일반 BN 접기 규칙(§3.3)만 적용하는 것으로 충분한지. GroupNorm·LayerNorm은 fp32로 남긴다.
- **Q4 원 설정 비교값**: (i)의 "원 설정" 열을 이 실행의 재계산 O(제안; PBE는 B §1 기준 18/18 붕괴 예상)로 둘지, 원고의 11/18(다른 checkpoint·다른 라벨)로 둘지. 제안은 둘 다 표시하고 판정 비교는 O로 하는 것이다.
- **Q5 (iii) 읽기 규칙**: AUC 차 0.02 기준을 넣을지, 수치 보고만 할지.
- **Q6 오래된 FAILED.txt 6개 run**: results.json이 complete이므로 포함(제안). Table 1이 같은 run을 쓰는지는 `<workspace>/data/main` 복사본과 sha를 대조해 확인한다.
- **Q7 w4 per-tensor 비교의 추가 셀**: S-w4g와 S-w4c(2×2 분해)를 유지할지. 비용은 checkpoint당 2 pass다.

## 11. 위험·한계 (사전 기록)
- retention 라벨은 val, 일치도 점수는 test 8장이다. 같은 checkpoint이지만 영상 집합이 다르다(P3와 같은 한계, R1 minor 12). 이번에는 같은 run의 split을 쓴다.
- L80에서 O의 PBE 셀은 18/18이 예상되어 O 묶음의 PBE 판별력은 없다. AUC는 66 checkpoint 묶음에서 의미가 있다.
- 일반 BN 접기 규칙은 모델 구조에 따라 일부 BN을 남길 수 있다. 남은 BN 수를 모델별로 보고하며, 남은 BN은 배포 그래프 정의의 한계로 적는다.
- per-channel w4는 대부분 툴체인의 실제 int4 커널과 rounding·zero-point 처리가 다를 수 있다(fake-quant 측정이다). 장치 측정이 아니다.

---

## AMENDMENT 0 — 동결 시 기록 (측정 실행 전, 2026-10-16 15:39:02 KST)

**sha256 (서버 `<WORK>/qdeploy/`, 측정 전; 실행 스크립트는 `FROZEN_SHA256SUMS`로 매번 검사)**
| 파일 | sha256 |
|---|---|
| qdeploy.py (engine, 연산자, fusion 호출, 중단 규칙) | 644bd5eeba8c1ded4d61810da8c74ac1ab3909e0e504ce76674ba2c819e5b899 |
| qdeploy_analyze.py (판정, §7.3 selftest) | f93895df79de58d3e612f1dfeea314fef4329bfa3ca87663f8db4b2f0a6166e2 |
| run_qdeploy.sh | dc603e111460071e3598c26cf0578ac67eb34849d83df0c46d7ba853c07458d2 |
| run_decide.sh | 98605ac411f84e189bb4ee973d155ba6bf770f7cf6136df0356e415271d8f505 |
| f2_fuse.py (F2 사본, byte-identical) | a1ef95d3a5c0efefbb9ef2d18023189bf077863357ab0c61ed36f8875c08706f |
| f2b_fold_all.py (F2b 사본, byte-identical) | 9ad43a00f7c3a1e39109052b8eb60137eb2fb2801933967fe32c74577dfed074 |
| qfix/C/rec/qfix_a.py (record 연산자, 수정 없음) | b36df82ffd238ccd2fadee1b02ab7df016e1bcc6500fc2c865c6adbc52fa94c6 |
| FOLD_PAIRS.json | 69e196dd1b41ea99bbf1f3e3401d2e560ef7b010177de3472c747957da7fc1d5 |
| MANIFEST_330.sha256 (330 checkpoint.pth + 330 results.json) | 58714f2f280ea80153cabff571c13efbc425700967adc2c48b0147fad510c2b6 |
| ITEMS.json (330 item, 실행 순서) | cc16bb780d77da85991ea5e76c77e373bb009f37cf6f0e88f6677aaabf672c61 |

**해시 대조(§2.2)**: PBE 30 = stage A `qfix/A/out` ckpt_sha256과 30/30 일치. 베이스라인 300 checkpoint = 독립 사본 `<workspace>/fieldswitch/ckpt`과 300/300 일치(k1 60 포함). 베이스라인 model 소스 10개는 archive `code`(e0e2f07)·`code_v2`(fbe35ca)·`qfix/A/code` 사이 byte-identical(diff -rq, __pycache__ 제외). PBE는 e0e2f07 = `qfix/A/code`와 같다(code_v2와 다름, PBE 30 runs는 모두 e0e2f07). 영상 캐시 누락 0.

**Fold pair (§F.3, §F.8)**: 무작위 stats 단독 접기 상대오차 최대 1.4e-15(utanet), 탈락 후보 0. 접힘 수는 §F.3.

**합성 역검사(§7, 같은 함수)**: `SYNTHETIC ALL PASS` — 7.1 Conv+BN fp64 2.2e-15, PBE fusion 정상 1.0e-13 / bn_identity 누락 1.3e+02 / var² 2.4e+08, tinyunet 일반 접기 정상 9.7e-14 / var² 7.5e+14; 7.2 per-tensor w4 이상치 채널 외 0 비율 1.000, per-channel 상대오차 0.0713(≤1/14), ConvT 축 0.0714, a6 per-channel 상대오차 0.0161(≤1/62), l50 layer·channel zero 0.5000; 7.3 분기 2/18→artifact, 5/18→survives, AUC 1/0/0.542, 손 계산 0.375 [0, 0.75]; 7.4(PBE kvasir k1 plant) fp32 Dice 차 0, E1 9.2e-14, |ΔR D-w4| = |ΔR S-w4c| = 0.0000, per-tensor 0 비율 1.000. 상세는 `synthetic.log`(23b5b3b2…bd98), `synthetic.json`(274a70a9…7f36).
