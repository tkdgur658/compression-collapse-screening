# PREREG_SGR — 압축 붕괴의 라벨 없는 복구(R2 검증) 사전등록 (동결본)

작성 2026-10-17. 상태: **동결**(동결 시각과 sha는 서버 `sgr/FROZEN_SHA256SUMS`, `sgr/FROZEN_AT`). 코디네이터가 §13의 ①–⑨를 원안대로 승인했고 (A)(B)(C)를 추가했다. 동결 시점에 복구 결과는 없다(GPU 미사용, 읽은 값 없음).
근거: `DESIGN_REPAIR.md`(SGR 설계), `SGR_FEAS.md`(CPU 점검 1 checkpoint), `G0p_R2_SGR.md`(선행 점검), 서버 F3 `qdeploy/PREREG_QDEPLOY_FROZEN.md`(sha 573b72eb…6d96), `qdeploy_tau/TAU_RESULT.json`(sha 8c7c55e0…b4f157).
코디네이터 결정(2026-10-17)을 반영했다.

**배포 형태 제약의 정의 (A).**
- 보존하는 것: 0이 아닌 가중치의 총수(전역 희소도 50 %)와 비트 폭.
- 바뀌어도 되는 것: 층별 희소도.
- 툴체인이 층별 희소도를 고정한다면 절제 A1("재배분 없음")이 그 조건 아래의 SGR이다. A1 결과는 그렇게 이름 붙여 보고한다.

**결과 공개 (C).** 중간 보고는 없다. decide 스크립트는 ALL_DONE 뒤 한 번만 돈다.

---

## 1. 질문과 가설
질문: F3 배포 그래프(RepConv fused, BN folded)에서 층별 50 % 크기 가지치기(D_l50)로 붕괴한 셀을, 배포 형태(마스크·전역 희소도 50 %)를 유지한 채 **라벨 없는 TRAIN 영상 n = 100장과 fp32 출력만으로** 되돌릴 수 있는가?

- **판정 단위 = 셀(모델 × 데이터셋).** 셀은 **복구됨** iff 복구 뒤 5-split 평균 retention ≥ 0.8이다. 이는 원 붕괴 라벨(F3 L80, 5-split 평균 val-Dice retention < 0.8)과 같은 정의다.
- **H_R1**: SGR 복구 셀 수 ≥ 0.7 × 55, 즉 **≥ 39/55**이면 PASS. 문턱 판정이며 검정은 없다. Holm에 넣지 않는다.
- **H_R2**: SGR 복구율 > B1 복구율. 55셀 짝지은 **정확 McNemar, 단측**(SGR 우위 방향)으로 검정한다.
  - b = SGR만 복구한 셀 수, c = B1만 복구한 셀 수다. p = P(Binom(b + c, ½) ≥ b)이다.
  - b + c = 0이면 p = 1로 둔다.
- **H_R4**(3모델 부분집합 13셀): SGR 복구 셀 수 ≥ 0.8 × B4 복구 셀 수.
  - 검정: 셀 단위 짝 bootstrap(B = 10000, seed 20261017, 13셀 복원추출)을 쓴다. p = Pr*(n_SGR* − 0.8 n_B4* ≤ 0)이다.
  - 점추정이 조건을 만족하고 Holm 조정 p < 0.05이면 PASS다.
  - B4 복구가 0셀이면 NOT TESTABLE로 두고 p = 1로 처리한다.
- **Holm**: {H_R2, H_R4}, α = 0.05, m = 2.
- **H_R3는 가설이 아니다.** 수정 파라미터 비율은 구성상 정해진다. PBE PH2 k1 CPU 점검에서 SGR 15,735(추론 파라미터의 1.2 %), B1 660,898(50.1 %)이었다. 실행에서 checkpoint마다 다시 계산하고 모델별 [min, max]를 보고한다.
- **예측**(DESIGN_REPAIR §3): H_R1 .55, H_R2 .45, H_R4 .4.
- **실패 분기**(DESIGN_REPAIR §3 그대로):
  - H_R1 미달이면 "마스크 보존 하 라벨 없는 복구는 x/55까지"를 사실로 보고한다. R2는 "부분 검증"이 된다.
  - B1 복구 셀 수 > SGR이면 SGR을 폐기하고 B1을 R2 구현으로 보고한다.

## 2. 자산: checkpoint 목록(규칙으로 생성, 완료)
`<WORK>/sgr/sgr_prep.py`(sha256 a79a57df…7345)로 생성했다. 동결된 F3 행(`qdeploy/out/items`)과 `qdeploy_analyze.inclusion / ckpt_rows / cell_table / counts`를 바꾸지 않고 썼다. 출력은 `ITEMS_SGR.json`(sha256 27398e53…c56e)이다. R 값은 출력하지 않고 소속만 썼다.

| 블록 | 규칙 | 수 |
|---|---|---|
| main | D_l50 붕괴 55/66 셀의 5 split 전부 | **275** checkpoint |
| sub | main ∩ {PBE-Net, MOST, FEWEST} | 65 (PBE 6셀, UTANet 6셀, MAResUNet 1셀(etis)) |
| w4 | S_w4g 붕괴 셀 ∩ {PBE-Net, MOST, FEWEST}, 5 split | 75 (PBE 6, UTANet 6, MAResUNet 3: etis·isic17·isic18) |
| naux | main ∩ PBE-Net | 30 |

- MOST / FEWEST는 D_l50 붕괴 셀이 가장 많은 / 적은 베이스라인이다. 동률이면 qdeploy `BASELINES` 순서를 따른다.
  - MOST = **UTANet**(6셀; maresunet 다음 순서로 utanet이 6셀 동률 모델 중 첫째).
  - FEWEST = **MAResUNet**(1셀).
- F3 제외 1건(malunet isic18 k4, old FAILED.txt 규칙)은 어느 붕괴 셀에도 속하지 않는다.
- 참고(판정 무관): main 55셀 중 split-1 checkpoint 자체가 R ≥ 0.8인 셀이 5개 있다(mhorunet/kvasir, mk_unet/isic17, ucmnet/isic17, ucmnet/isic18, utanet/PH2). checkpoint 단위 D_l50 붕괴는 271/329다.

## 3. 영상 사용 (F3와 일치, test 봉인 유지)
| 양 | 영상 | 라벨 | 근거 |
|---|---|---|---|
| retention R = Dice_q / Dice_fp32 | 그 run의 **val** 캐시 전체(저장 순서) | val masks | F3 §(91–92). 분모 = 배포 그래프 fp32 val Dice(이 실행에서 재계산) |
| 재선별 agreement(SGR 4단계 보고) | 그 run의 **test** 캐시 `images` 앞 8장 | 미접근 | F3 §(94), P3 방식. fp32 기준 = 배포 그래프 |
| 복구 입력(진단·재배분·정합, B1, B3, 절제) | 그 run의 **train** 캐시 `images`, 고정 시드로 뽑은 n = 100장 | **미접근** | 코디네이터 결정. val·test 미사용 |
| B4 (라벨 상한) | 위와 **같은** train 100장 | **train masks** | 코디네이터 결정 |

- 복구 단계는 val·test 영상을 보지 않는다. val은 R 계산에만, test 8장은 agreement 재선별 보고에만 쓴다. 이 둘은 F3에서 쓴 영상과 같다.
- **정정 기록 (B).** 코디네이터 문구 "train-image-based retention"은 F3 정의로 바로잡았다. 판정 retention = val Dice(압축) / val Dice(fp32)이며 val 라벨을 쓴다(F3 그대로).
- **선택 풀 (B, 동결 전 점검으로 추가한 규칙).** train 영상 가운데, 같은 run의 val 또는 test에 파일명이나 영상 바이트 sha256이 같은 영상이 있으면 풀에서 뺀다.
  - 첫 점검(`sgr/OVERLAP.json`, 보존): 파일명 중복은 0이었다. 그러나 내용 중복이 285 checkpoint 합계 train 62장 있었다(etis k2·k4·k5, isic17 k2·k4). 원래 규칙으로 뽑은 100장 중 18장이 val, 7장이 test와 내용이 같았다.
  - 재점검(`sgr/OVERLAP_dedup.json`, `sgr_overlap.py`): 선택 100장(진단 20장 포함)은 285 checkpoint 전부에서 val·test와 파일명·해시 중복 0이다(**PASS**). 제외된 62장은 기록했다.
  - 실행 중 Job도 같은 검사를 다시 하고, 0이 아니면 assert로 멈춘다.
- 시드(checkpoint별 고정): seed_ck = (int(sha256(tag)[:8], 16) ^ 20261017) mod 2³¹.
  - perm = 풀 index를 `np.random.RandomState(seed_ck).permutation(|풀|)` 순서로 늘어놓은 것이다(`sgr_run.train_order`). n = 100은 perm[:100]이다.
  - 진단용 20장 = perm[:20]이다. naux n = 10, 30은 perm[:10], perm[:30]이고, 진단은 perm[:min(n, 20)]이다.
  - 모든 데이터셋에서 split 1의 train 크기가 100장 이상이다(etis 158, PH2 120 …).

## 4. 그래프와 압축 연산자 (F3 코드 재사용, 변경 없음)
- 배포 그래프: `qdeploy.Item`을 그대로 쓴다. PBE는 `f2_fuse.fuse_pbe`, 베이스라인은 `FOLD_PAIRS.json` 쌍으로 접는다. E1 fp64 등가성 검사도 Item 그대로 수행한다.
- 정밀도: fp32, TF32 off, cudnn deterministic, eval 배치 16, 임계 σ > 0.5. Dice 관례는 `Item.val_metrics`와 같다.
- D_l50(B0의 마스크): `Graph.apply('l50')`, 즉 실행된 Conv/Linear(weight dim ≥ 2)마다 `qfix_a.prune_layer`(|w| ≤ k번째 작은 값 → 0, k = ⌊0.5 n⌋)를 적용한다.
- S_w4g(w4 블록): `Graph.apply('w4_pt')`(배포 그래프, dim ≥ 2 텐서마다 per-tensor 대칭 RTN 4비트). bias는 fp32로 남는다.
- 층 집합 𝓛: 실행된 l50 층이다(PBE 37층: Conv 16, depthwise 12, Linear 9; F3 integrity의 n_layers와 같다).

## 5. 팔(arm) 정의
공통 사항:
- 마스크 M_l은 각 팔의 가지치기 결과이며 고정이다. 가중치는 W_l ⊙ M_l ⊙ γ_l로 매개변수화한다(γ_l = 출력 채널별 배율, 초기 1, 내보낼 때 W에 다시 접음).
- 가지친 원소는 정확히 0이고 기울기가 0이다. 내보낸 뒤 0의 수가 마스크와 같은지 assert한다.
- bias는 그 층에 bias가 있을 때만 조정한다. 이는 접힌 BN의 shift이며, PBE는 16/37층이 bias를 가진다. bias가 없는 층에는 bias를 새로 만들지 않는다.

| 팔 | 마스크 | 조정 대상 | 손실 / 절차 |
|---|---|---|---|
| **B0** | record l50 | 없음 | — (F3 D_l50 재현 검사, §8) |
| **B1** 전역 KD | record l50 | 모든 층의 남은 W + γ + bias | 정합 손실 L_match, 300 스텝 |
| **B2** 전역 크기 가지치기 | 𝓛 전체에 하나의 |w| 임계. 정확히 ⌊0.5 N⌋개를 0으로(동률은 평탄 index 순) | 없음 | — |
| **B3** 배포형 채널 통계 보정 | record l50 | 층마다 γ(전 층), bias(있는 층) | 아래 정의. 비반복(경사 스텝 없음) |
| **B4** 라벨 상한 (sub만) | record l50 | B1과 같음 | L_label = 1 − softDice(σ(z), y) + BCE(z, y), y = train masks, 300 스텝 |
| **B5** 채널별 재양자화 (w4만) | — | 없음 | `qdeploy.rtn_sym_channel` 4비트 = F3 D_w4 연산자 |
| **SGR** | 재배분 마스크 | L\*의 남은 W + 전 층 γ + bias | 1–4단계(아래) |
| A1 재배분 없음 (sub) | record l50 | SGR과 같음 | 1, 3단계 |
| A2 정합 없음 (sub) | 재배분 마스크 | 없음 | 1, 2단계 |
| A3 무작위 층 (sub) | 무작위 층으로 재배분 | 무작위 층의 W + γ + bias | 1단계로 \|L\*\|와 귀속 크기를 얻는다. 같은 수의 층을 무작위(seed_ck + 1)로 뽑고, L\*의 귀속값을 무작위 순서로 배정한 뒤 2, 3단계를 수행한다 |

**정합 손실** L_match = 1 − mean_i softDice(σ(z_c,i), σ(z_fp,i)) + mean MSE(z_c, z_fp). softDice(p, q) = (2Σpq + 1e-6) / (Σp + Σq + 1e-6), 영상별로 계산한다. z_fp = fp32 배포 그래프 logit(100장, 미리 계산).

**B3 (접힌 BN의 배포형 대응, 라벨 없음, 순차).** 𝓛을 forward 실행 순서로 정렬하고, l = 1…|𝓛|마다 다음을 수행한다.
- 앞선 층이 이미 보정된 압축 그래프와 fp32 배포 그래프를 n = 100 train 영상에 통과시켜, 층 l 출력의 채널별 통계를 얻는다. 채널 축은 Conv dim 1, Linear는 마지막 dim이고, 영상·공간 위치 전체에 대한 모집단 통계다. 이로부터 μ^c, σ^c(압축)와 μ^fp, σ^fp(fp32)를 구한다.
- bias가 있는 층: γ = clip(σ^fp / max(σ^c, 1e-8), 0.1, 10), W ← γ ⊙ W, b ← γ (b − μ^c) + μ^fp. 출력 평균과 분산을 fp32와 맞춘다.
- bias가 없는 층: γ = clip(RMS^fp / max(RMS^c, 1e-8), 0.1, 10), W ← γ ⊙ W. 배율만 맞춘다.
- 마스크는 γ가 채널 배율이므로 보존된다.
- 출처: EagleEye 적응 BN(Li 외 ECCV 2020)의 통계 재추정과 DFQ bias correction(Nagel 외 ICCV 2019)을 "apply"한다. BN 원래 (β, |γ|)를 목표로 삼지 않는 이유는 PBE RepConv처럼 BN 세 개가 한 conv로 합쳐진 층에서는 그 목표가 정의되지 않기 때문이다.
- 비용은 |𝓛| × 100장 forward다(§10).

**SGR 1–4단계 (D_l50).**
1. **진단**: record l50 상태에서 층 l ∈ 𝓛 하나만 fp32 가중치로 되돌린다. 20장에서 D_soft = 1 − mean softDice(σ(z), σ(z_fp))와 D_hard = 1 − mean `Item.pair_dice`(임계 마스크)를 계산한다. 이득 g_l = D_soft(전부 l50) − D_soft(l만 fp32)이다.
   - L\* = g_l > 0인 층을 g_l 내림차순(동률은 D_hard 이득, 그다음 이름)으로 정렬했을 때 누적합이 Σ_{g>0} g의 80 %에 처음 도달하는 최소 접두사다.
   - g > 0인 층이 없으면 L\* = argmax g다.
2. **재배분**: a_l = g_l / max g for l ∈ L\*, 그 밖은 0이다. s_l = clip(t − c a_l, 0, 0.9), c = 0.5다. t는 이분법으로 정해 Σ s_l n_l = 0.5 N(파라미터 가중)을 맞춘다.
   - 정수 개수 k_l = round(s_l n_l)로 하고, 잔차는 여유가 있는 큰 층부터 배정한다. 그러면 Σ k_l = ⌊0.5 N⌋이 정확히 성립한다.
   - 층마다 fp32 |w| 하위 k_l개를 0으로 한다(동률은 index 순).
   - 전역 희소도 0.5 ± 0.001을 assert한다(CPU 점검: 0.50000).
3. **국소 정합**: L\*의 남은 W, 전 층 γ, bias를 L_match로 300 스텝 조정한다. 마스크는 고정한다.
4. **재선별**: test 8장 agreement를 F3 방식으로 계산한다. τ_score = 0.26212042899165(TAU_RESULT L80, agreement 0.7379)를 쓰며, 1 − agree ≥ τ_score이면 "flag"다.
   - 이 단계는 **보고만** 하고 셀 판정을 바꾸지 않는다.
   - flag가 남으면 "마스크·눈금 변경 없이 복구 불가"로 기록한다.

**SGR-w4 (w4 블록, 보고만).**
- 1단계: 단위 = 양자화 텐서(dim ≥ 2)다. 하나만 fp32로 되돌려 이득을 구한다.
- 재배분 단계는 없다(코디네이터 결정).
- 3단계: L\*_q 텐서의 잠재 fp 가중치를 STE로 조정한다. 매 forward에서 `rtn_sym_tensor` 4비트로 양자화하고, 눈금 = max|w|/7(F3와 같다)이다. 전 층 bias도 함께 조정한다.
  - γ는 쓰지 않는다. 채널 배율이 per-tensor 형태를 깨기 때문이다.
  - 끝나면 최종 재양자화하고 텐서당 수준 ≤ 15를 assert한다.
- w4 블록의 B1 = 모든 잠재 가중치 STE + bias이고, B3 = bias 보정만(위 식에서 γ = 1)이다. B0, B5는 위와 같다. B2와 절제는 없다.

## 6. 고정 설정 (모든 팔 공통, 지금 고정)
- 스텝 300(B1, B4, SGR 3단계, A1, A3, SGR-w4, B1-w4). B3는 비반복이므로 스텝 예산이 해당되지 않는다.
- 배치 8. 스텝마다 100장에서 복원추출(torch.Generator(seed_ck))하며, 증강은 없다.
  - 근거: 3090 24 GB에서 UTANet(24 M)·MAResUNet(26 M)의 256² backward가 들어가는 크기다. 300 × 8 = 2400 영상-스텝으로 n = 100을 약 24 epoch 본다.
- 옵티마이저 Adam(β = 0.9, 0.999), **lr 1e-4**, cosine decay → 0(300 스텝), weight decay 0, gradient clipping 없음. 모든 조정 텐서(W, γ, bias)에 같은 lr을 쓴다.
  - 근거: 재구성 기반 PTQ·가지치기 미세조정(AdaQuant, EBFT)에서 쓰는 1e-4–1e-3 범위의 보수적 끝이다. 전 가중치를 움직이는 B1이 fp32에서 멀어지지 않게 하는 값이며, 팔 간 차이가 lr 차이에서 나오지 않게 한다.
- 모델은 eval 모드(드롭아웃 off)이고, 배포 그래프에는 BN이 없다.
- 첫 OOM이 나면 그 checkpoint를 배치 4, 같은 300 스텝으로 재시도하고 기록한다. 두 번째 OOM이면 정지한다.
- n = 100이 주 설정이다. naux(PBE 30 checkpoint)에서는 SGR과 B1을 n = 10, 30으로도 수행하며 보조로만 쓴다.

## 7. 지표와 보고
- 주: 팔별 복구 셀 수 /55(main), /13(sub), /15(w4 보고)와 H_R1, H_R2, H_R4 판정.
- 보조:
  - checkpoint 단위 표: 팔별 R ≥ 0.8 수 /275.
  - 재선별 flag 수, 수정 파라미터 비율(모델별 [min, max]).
  - fp32 대비 agreement, |L\*|와 L\* 파라미터 비율, 재배분 뒤 층별 희소도 범위.
  - n = 10/30/100 곡선(PBE), w4 블록 결과.
- 셀 평균은 F3 `cell_table`과 같은 규칙을 쓴다. 판정 가능한 split만 평균하고 split 수를 보고한다. FAIL로 split이 빠진 셀은 표에 표시한다.
- 블록이 미완이면(정지) 그 블록에 의존하는 가설은 NOT TESTABLE이다.

## 8. 무결성 검사 (실행 중)
- 매 checkpoint에서 다음을 검사한다.
  - `MANIFEST_330.sha256` 일치. 불일치 시 정지한다(코디네이터 결정, PBE·베이스라인 모두).
  - E1 fp64 등가성(F3 Item).
  - **B0 val Dice가 F3 D_l50 기록과 |Δ| ≤ 1e-6**(같은 코드·GPU). 어긋나면 모든 모델에서 정지한다(STOPPED_REPRO, 코디네이터 결정).
  - fp32 배포 val Dice가 F3 기록과 |Δ| ≤ 1e-6이어야 한다. 어긋나면 STOPPED_REPRO.
  - E1 실패 시 STOPPED_E1, 상태 복원 실패 시 STOPPED_STATE.
  - w4 블록의 B0w(S_w4g)와 B5(D_w4)의 F3 재현 차이는 기록만 한다.
- 팔마다 다음을 확인한다.
  - 전역 희소도(B2·SGR·A1–A3: 정확히 ⌊0.5 N⌋/N, B0·B1·B3·B4: record l50과 같은 0의 수).
  - 내보낸 뒤 마스크 보존(0의 수 동일), 유한 logit.
  - w4 수준 ≤ 15.
  - 상태 복원 정확성(`Graph.exact`).
- 로그에는 시간과 무결성 카운터만 남긴다. Dice와 R은 JSON에만 봉인하고 decide 전에 출력하지 않는다.

## 9. 합성 역검사 (실제 실행 전, CPU)
1. **결정 스크립트**(`sgr_decide.py --selftest`)를 실제와 같은 집계 함수에 넣는다. 가짜 checkpoint R 4개 시나리오와 기대 판정은 다음과 같다. 각 시나리오에는 한 split < 0.8이지만 5-split 평균 ≥ 0.8인 셀을 하나 넣어 평균 규칙도 검사한다.

| 시나리오 | 가짜 값 | 기대 |
|---|---|---|
| S1 전부 복구 | 모든 팔 R = 0.95 | H_R1 PASS, H_R2 FAIL(b = c = 0, p = 1), H_R4 PASS(p = 0) |
| S2 전부 실패 | 모든 팔 R = 0.3 | H_R1 FAIL, H_R2 FAIL, H_R4 NOT TESTABLE(B4 = 0) |
| S3 B1 우위 | B1 50셀, SGR 30셀(SGR ⊂ B1), sub에서 SGR 6, B4 13 | H_R1 FAIL, H_R2 FAIL, 분기 "SGR 폐기·B1 보고", H_R4 FAIL |
| S4 SGR 우위 | SGR 45셀(sub 13셀 포함), B1 20셀(B1 ⊂ SGR, b = 25, c = 0), B4 13 | H_R1 PASS, H_R2 PASS(p = 2⁻²⁵), H_R4 PASS, Holm 둘 다 PASS |

2. 단위 검사(CPU, 합성 텐서와 PBE PH2 k1 train 영상만 사용, Dice 미계산):
   - 재배분 전역 희소도가 정확한지.
   - 3 스텝 뒤 마스크가 보존되는지.
   - B3 보정 뒤 채널 통계가 목표와 상대 1e-4 이내로 맞는지(toy conv망).
   - STE w4 수준 ≤ 15.
   - McNemar 기준값: b = 10, c = 2 → p = 79/4096 = 0.019287.
   - Holm 기준값.
- 하나라도 기대와 다르면 동결하지 않는다.

**동결 전 실행 결과.**
- `sgr_decide.py --selftest`: 4 시나리오, 평균 규칙(straddle 셀 cmunext|PH2), McNemar 기준값, Holm 기준값 모두 **PASS**(`SELFTEST.json`, `synthetic.log`). S4의 p = 2⁻²⁵ = 2.98e-8이다.
- `sgr_run.py --unit`: 재배분 정확(148/148), 전역 크기 정확, 마스크 보존, B3 통계 일치(상대오차 6.4e-7, 클립 0), STE 수준 ≤ 15 모두 **PASS**(`unit.json`).
- CPU smoke(`smoke_all.sh`, `logs/smoke.log`): 11 모델 etis k1 checkpoint로 main 블록을, PBE·UTANet·MAResUNet으로는 4 블록 전부를 돌렸다. 팔마다 2 스텝, train 8장이며 Dice는 계산하지 않았다. 결과: 20 레코드 무결성 위반 0(`smoke_check.py smoke`).
- w4 학습 전용 텐서를 뺀 뒤 w4 블록(3모델)과 PBE main을 다시 돌렸다(`smoke2/`): 4 레코드 위반 0.
  - 검사 항목: 유한 출력, 마스크 보존, 정확한 0 개수, 전역 희소도 0.5, w4 수준 ≤ 15, 2 스텝 완료, 상태 복원.

## 10. 구현 계획
서버 폴더 `<WORK>/sgr/`(신규). archive는 읽기만 하며 PYTHONDONTWRITEBYTECODE=1로 실행한다.
| 파일 | 내용 |
|---|---|
| `sgr_prep.py` (완료) | §2 목록 → `ITEMS_SGR.json` |
| `sgr_lib.py` | exact_prune, realloc_sparsity, global_magnitude_counts, MaskScale, STE w4, B3 보정, 손실, 진단(`sgr_feas.py`에서 이전) |
| `sgr_run.py` | `--unit`(§9.2, CPU), `--smoke`(CPU, 1 checkpoint, 팔마다 2 스텝, 무결성만 출력), `--run`(GPU, 큐). 큐 단위 = (tag, block)이고, 산출은 `out/items/<tag>__<block>.json`과 `.DONE / .FAIL / .OOM`이다 |
| `sgr_decide.py` | `--selftest`(§9.1), `--decide`(1회) → `RESULT_SGR.json / .md` |
| `run_sgr.sh gpu0` / `run_sgr.sh gpu1` | worker. `FROZEN_SHA256SUMS` 검사, worker별 flock, CUDA_VISIBLE_DEVICES = 0 또는 1, threads 1. 외부 PID 대기(qdeploy `foreign_pids` 방식). DONE·FAIL·OOM 시간 합으로 cap을 관리한다. gpu1은 `<WORK>/abl/ALL_DONE`이 생길 때까지 300 s 간격으로 기다리며, 그동안 GPU를 쓰지 않는다 |
| `sgr_overlap.py` | §3 선택 풀의 val·test 중복 검사 → `OVERLAP_dedup.json` |
| `run_decide.sh` | ALL_DONE 없거나 RESULT 있거나 DECIDE_STARTED 있으면 거부, sha 검사(F3 run_decide.sh와 같은 구조) |

- **실행 순서(우선순위)**: ① main 275(PBE → BASELINES 순, ds → k) ② sub 추가 팔 65 ③ w4 75 ④ naux 30.
  - cap으로 멈춰도 H_R1·H_R2가 먼저 판정 가능하도록 한 순서다.
  - GPU1은 F4 종료 뒤에만 쓴다(`abl/ALL_DONE`).
  - 두 worker는 같은 순서의 큐에서 다음 미완 단위를 가져간다. claim은 `out/queue.lock` flock 아래 `.CLAIM` 파일로 한다. 죽은 worker의 claim은 pid가 살아 있지 않을 때만 다시 가져간다.
  - 블록 순서는 (C) 그대로 main → sub → w4 → naux다.
- **정지 규칙**: compute 20 GPU-h 도달 → STOPPED_CAP. 두 번째 OOM → STOPPED_OOM. 다른 사용자 프로세스가 GPU에 있으면 60 s 간격으로 대기. manifest / sha 불일치 → STOPPED_SHA. B0 또는 fp32 재현 차이 > 1e-6(모든 모델) → STOPPED_REPRO. 배치 4 재시도에서도 OOM이면 즉시 STOPPED_OOM.
  - ALL_DONE은 두 경우에만 쓴다. 모든 단위가 DONE 또는 FAIL이 된 경우, 그리고 STOPPED_CAP만 있고 살아 있는 claim이 없는 경우다. 후자는 `partial: true`와 미완 블록 목록을 함께 적는다.
  - 다른 정지(OOM, SHA, REPRO, E1, STATE, FAILS)에서는 ALL_DONE을 쓰지 않는다. 원인을 확인하는 것은 사람의 몫이다.
  - 동결 전 추가한 정지 규칙: 한 worker에서 FAIL이 3번 연속 나면 STOPPED_FAILS다. 체계적 버그 하나가 큐 전체를 FAIL로 채우는 것을 막기 위해서다.
- **비용 추정(backward 포함)**:
  - CPU 측정(`sgr_feas/measure_step.py`, sha 3e589a14…0b3, bs8, 2 thread)으로 masked fwd + bwd + Adam 1 스텝을 8 × eval forward(영상당)로 나눈 비율 ρ를 구했다. PBE 2.23, UTANet 2.91, MAResUNet 3.34, TinyUNet 5.16, EGE-UNet 3.10, MHorUNet 3.86이다. 측정하지 않은 4모델에는 최대값 5.16을 쓴다.
  - **가정**: GPU 스텝 = ρ × 8 × f_m이다. f_m은 F3 run.log 회귀(checkpoint 시간 대 n_val의 기울기 ÷ 12 pass)로 얻은 3090 영상당 forward 시간이다(PBE 3.58 ms, UTANet 8.30 ms, egeunet 0.61 ms). 즉 GPU의 backward/forward 비가 CPU와 같다고 가정한다.
  - 포함 항목: 로드, 목표 logit 100장, 팔마다 eval(val + test 8), SGR 진단 (|𝓛| + 1) × 20장, B3 |𝓛| × 100장.

| 블록 | 300 스텝 | 150 스텝(참고) |
|---|---|---|
| main 275 × B0, B1, B2, B3, SGR | 6.1 GPU-h | 4.2 |
| sub 65 × B4, A1, A2, A3 | 2.2 | 1.2 |
| w4 75 × B0, B1, B3, B5, SGR-w4 | 2.4 | 1.6 |
| naux 30 × (SGR, B1) × n = 10, 30 | 0.7 | 0.4 |
| **합** | **11.4** | 7.4 |

  - cap 20 GPU-h 안에 들어가지만, 여유는 1.75배뿐이다. F3 기울기에는 a6 hook과 val 지표 계산이 들어 있어 f_m은 과대 추정 쪽이다.
  - GPU0 단독 벽시계는 약 11–12 h, F4 뒤 GPU1을 더하면 약 7 h다.
  - 순서 ①만으로 6.1 GPU-h이므로, cap에 걸려도 주 가설은 판정 가능하다.

## 11. F3 정의와의 차이·충돌 (명시)
1. **진단 영상 ≠ 재선별 영상.** 진단 D는 train 20장에서 계산한다. R1 재선별(4단계)과 F3 agreement는 test 8장을 쓴다. 함수는 같고(`pair_dice`, 같은 임계) 영상 집합이 다르다. test는 복구에 쓰지 않는다.
2. **같은 τ를 같은 test 8장에 다시 쓴다.** τ는 F3 split-1 행에서 Youden으로 고른 사후 값이다(`tau_youden.py`, 사전등록 밖). 그래서 재선별은 보고만 하고 판정에 쓰지 않는다.
3. **가지치기 동률 규칙.** record l50(`prune_layer`)은 |w| ≤ 임계를 모두 0으로 한다(동률 시 k 초과 가능). B2·SGR·A1–A3는 정확히 k개로 한다. PBE에서 record l50 전역 희소도는 0.50000이었다. 작은 층(n = 3)에서는 F3 min_zero_frac가 0.333이었다.
4. **DESIGN_REPAIR와의 차이.**
   - (a) "텐서별 4비트 38/66"은 F3의 **S_w4g**(민감도 집합 S)다. D_w4(per-channel)는 9/66이다.
   - (b) H_R3(≤ 20 %)는 구성상 성립하므로 가설에서 뺐다.
   - (c) DESIGN의 "셀당 25분, 23 GPU-h"는 이 범위(275 checkpoint, B3 순차 포함)에서 11.4 GPU-h로 바뀐다.
   - (d) DESIGN은 "BN 접힘 scale/shift"를 조정한다고 썼다. 접힌 그래프에서 실제로 조정할 수 있는 것은 다음 둘이다.
     - bias가 있는 층의 bias(PBE 16/37).
     - 새로 붙여 다시 접는 채널 배율 γ. BN scale은 W 속으로 사라졌다.
5. **B3 이름.** 배포 그래프에는 BN이 없다(PBE 33 → 0). 그래서 B3는 BN 재추정 그 자체가 아니라 §5의 채널 통계 보정이다.
6. **배포 형태.** SGR·B2는 층별 희소도를 바꾸고 전역 50 %만 유지한다. 층별 50 %를 요구하는 툴체인이라면 이 둘은 배포 형태 위반이다.
7. **현장성 한계 (C).** 원고 한계 문장: "The unlabeled images used for repair are training images of the same split, not field images, so the repair rates are optimistic for deployment data." 무라벨 영상은 학습에 쓰인 분포 내 영상이며, DESIGN §0의 "현장 영상"과 다르다.
8. **선택 풀 규칙 추가.** §3의 내용 중복 제외는 동결 전 점검에서 생긴 규칙이다. F3에는 이 규칙이 없다. F3의 val·test 지표는 그대로 두며, 원래 split 안의 중복은 F3 학습 때부터 있던 것이다.

## 12. 공개(disclosure) — 원고 서술 규칙 (G0p_R2_SGR §6–7)
- SGR은 노벨티 주장이 아니다. 원고에는 R2 규칙의 **검증된 구현**으로만 "We apply …" 문장으로 쓰고, 조합 자체를 기여로 주장하지 않는다.
- 단계별 출처:
  - 1단계: 한 층씩 되돌려 무라벨 불일치로 민감도를 매긴다 → Wu 외 2020(arXiv:2004.09602), **ZeroQ**(Cai 외 CVPR 2020).
  - 2단계: 전역 희소도 고정 하 비균일 재배분 → **SPDY**(Frantar·Alistarh ICML 2022), BESA(ICLR 2024), OWL(ICML 2024), LAMP(ICLR 2021).
  - 3단계: 마스크 보존 출력 재구성 → **EBFT**(arXiv:2402.12419), **AdaQuant/AdaPrune**(Hubara 외 ICML 2021 / NeurIPS 2021), FSKD(CVPR 2020), PD-Quant(CVPR 2023).
  - 4단계·고리: 탐지 → 복구 → 재검사 → **CompD**(Li 외 ACML 2023, PMLR 222), DFLARE(TOSEM 2023).
  - 기준선: B1 = Polino 외 ICLR 2018. B3 = EagleEye(ECCV 2020) + DFQ bias correction(Nagel 외 ICCV 2019, 이번에 추가하며 원문 미확인). AdaRound, BRECQ.
- 성공 시 문장 틀은 DESIGN_REPAIR §5를 따른다. 실패 시에도 같은 형식의 음의 결과로 쓴다.

## 13. 설계 선택 (코디네이터 승인 2026-10-17: ①–⑨ 원안, ⑩ 시드 규칙 포함)
1. lr 1e-4, cosine, 모든 텐서에 같은 lr(γ·bias에 더 큰 lr을 쓰지 않음).
2. 재배분 계수 c = 0.5(최상위 귀속 층은 희소도 0, 즉 dense), s_max = 0.9, 귀속은 L\* 안에서만. CPU 점검 때는 c = 0.4였다.
3. L\*는 D_soft 이득 기준이다(20장에서 D_hard가 거칠기 때문).
4. B3는 순차 통계 보정이고 목표는 fp32 통계다(BN (β, |γ|) 목표가 아님). bias 없는 층은 배율만 맞춘다. γ 클립은 [0.1, 10]이다.
5. B4 손실은 softDice + BCE(라벨)이고, 조정 범위는 B1과 같다.
6. H_R4 검정은 셀 짝 bootstrap이다. 13셀이라 검정력이 낮다.
7. SGR-w4 3단계: STE 잠재 가중치 + bias, γ 없음. w4의 B3는 bias만 보정한다.
8. OOM 재시도는 배치 4, 같은 300 스텝이다(본 영상 수는 절반).
9. naux에서 B1도 n = 10/30으로 돌린다(+0.35 GPU-h).
10. checkpoint별 시드 = sha(tag) 기반.

## 14. 동결 절차
§13을 확정했고 §9의 검사를 통과한 뒤 동결했다. 사본은 `prereg/original/08_PREREG_SGR.md`에 있다.
- w4 블록에서는 학습 전용 텐서(adapter `is_train_only`)를 진단과 조정에서 뺀다. B0w·B5는 F3 연산자를 그대로 쓴다.
- 진단 결과는 Job 안에서 n_img별로 캐시한다. 그래서 SGR, A1, A2, A3는 같은 L\*를 쓴다.
- `FROZEN_SHA256SUMS`에 다음을 적는다: 이 문서, sgr_prep.py, ITEMS_SGR.json, sgr_lib.py, sgr_run.py, sgr_decide.py, run_sgr.sh, run_decide.sh, 그리고 참조하는 F3 파일(qdeploy.py 644bd5ee…, qdeploy_analyze.py f93895df…, f2_fuse.py, f2b_fold_all.py, FOLD_PAIRS.json, MANIFEST_330.sha256, ITEMS.json, qfix_a.py b36df82f…, TAU_RESULT.json 8c7c55e0…).
- 동결 뒤 이 문서의 수정은 별도 정정 절로만 한다. 결과는 decide 1회로만 연다.

| 동결 파일 (서버 `sgr/`) | sha256 |
|---|---|
| sgr_prep.py | a79a57df89e86991d63a47bcd51357d239c28eed5a0ce9b65183f08cfe6d7345 |
| ITEMS_SGR.json | 27398e53da6d11ceb18ac04792869db10906b51ce7b677095ff89721ab6fc56e |
| sgr_lib.py | 6abc19a8c756625714ad34184e46e2dc16c2f730d13c154f6f8a70c700ff1322 |
| sgr_run.py | 29a44ccb169b9b0cc0eaab894e1c16dfde767cf4e51124cda394f9c252c756a4 |
| sgr_decide.py | 77bf3244bd7af94920755c073189b8f8646218e567db78e2a3ff0c163f9a8177 |
| sgr_overlap.py | c834d082cc87a348d09e726fd92b996f01f1b0c56cb58f419a8a5e9f2425d0fa |
| run_sgr.sh | b35f24f6ab363f0b0c476295744975020be4db97d10d9e9306189e426197b1c6 |
| run_decide.sh | cab39ef4a3b6bef9d79b5996187cb9419781f751d7776479a02ce91f72ed5639 |
| OVERLAP_dedup.json | 191bb0eb7dae18e48ba64bff2d6bc8c25b45423dd5b35a78428cccc35f9d8ec5 |
| SELFTEST.json | f356ba2ff671031bb4519bea624352b166f586441188064ad2fbd14290ce0e63 |
| unit.json | 2d9137b962992dfcc82d162207a88555ee4b3d6e84d7fa4b465ecea458044c52 |

이 문서의 sha는 `FROZEN_SHA256SUMS`(PREREG_SGR_FROZEN.md)와 `prereg_release/SHA256SUMS.txt`에 있다.
