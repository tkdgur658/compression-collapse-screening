# B_fix_design — QFIX 단계 B (수정 설계), 2026-10-16

근거: `PREREG_QFIX.md` (sha256 1bac4879f7f56b235335f47d911f51a4aefc14febde7769adc4a79198fbd94ea, 확인 완료, 변경 없음) §2·§3, 단계 A 결과 `A_attribution.md/json`, `A_PLAN.md`, 스크립트 `A/qfix_a.py` (sha256 b36df82f…c6c6).
작성 조건: 로컬 읽기 + web 2회(G0′만). 서버·GPU·[internal review] 0. 측정 전 고정 문서이며 단계 C 결과를 본 뒤 수정하지 않는다.

**한 줄 요약.** H_A FAIL(top-2 share 0.654 < 0.70) → 사전등록 분기대로 단계 B = **B2 전체 compression-aware training만**(B1 표적 수정 없음). 기준선 재계산: **18셀 중 retention ≥0.8 셀 0/18**(checkpoint 단위 6/90). 기준 (i) "≥16/18"은 그대로. 비용 추정 C1 ≈ 14.7–17.9 GPU-h(중앙 ≈ 15.9) → 16 GPU-h 경계, 착수 전 timing gate 필요. 방법 기여 = 교과서 조합(중하 이하).

---

## 1. 기준선 재계산 (PREREG §0 정정, 기준 불변)

정의: 셀 = (dataset, setting ∈ {w4_pt, a6, l50}). 셀 값 = 5 split 평균 R_all, R_all = Dice_q / Dice_fp32 (val, 모든 구성요소 동시 적용 = record whole-network 설정). 셀 OK ⇔ 평균 ≥ 0.8. 출처 `A_attribution.json` (sha256 f1578292…d591)의 `cells[*].ret_all`. 출력 `B_baseline_18cells.json`.

| dataset | w4_pt 평균 (SD) [k≥0.8] | a6 평균 (SD) [k≥0.8] | l50 평균 (SD) [k≥0.8] | fp32 val Dice |
|---|---|---|---|---|
| PH2 | 0.644 (0.202) [1/5] | 0.679 (0.140) [2/5] | 0.135 (0.207) [0/5] | 0.947 |
| cvc | 0.288 (0.117) [0/5] | 0.105 (0.098) [0/5] | 0.013 (0.021) [0/5] | 0.905 |
| etis | 0.200 (0.177) [0/5] | 0.065 (0.051) [0/5] | 0.020 (0.041) [0/5] | 0.770 |
| isic17 | 0.642 (0.292) [2/5] | 0.423 (0.193) [0/5] | 0.006 (0.007) [0/5] | 0.898 |
| isic18 | 0.594 (0.265) [1/5] | 0.404 (0.211) [0/5] | 0.094 (0.140) [0/5] | 0.897 |
| kvasir | 0.158 (0.159) [0/5] | 0.123 (0.155) [0/5] | 0.023 (0.052) [0/5] | 0.866 |
| **OK 셀** | **0/6** | **0/6** | **0/6** | |

- **18셀 기준선: 0/18** (최대 셀 평균 = PH2 a6 0.679).
- 대안(checkpoint 단위): **6/90** healthy (w4 4/30, a6 2/30, l50 0/30) = 단계 A의 "84/90 붕괴"와 동일.
- 라벨 없는 판별기 기준값(v1 90셀, score = 1 − agree_all, label = R_all < 0.8, Mann–Whitney): **AUC = 0.978** (84 vs 6; 음성 6개뿐이라 불안정). §0의 .991은 P3 checkpoint에서 나온 값.
- **정정 범위**: §0의 서술적 기준선 "7/18 healthy (11 붕괴)"만 정정한다. §0 값은 P3 `queue10_pbe_reference` checkpoint(v1 runs와 sha 불일치)에서, 아마 chance-level 규칙으로 계산된 값이다. **성공 기준 (i) "retention ≥0.8인 셀 ≥16/18"과 셀 정의(dataset × setting, 5 split 평균)는 바뀌지 않는다.** 결과적으로 (i)이 요구하는 개선 폭은 사전등록 당시 생각한 +9셀이 아니라 +16셀이다. 따라서 §3의 "C 성공 .7" 예측은 낙관적이며, 이 문서 기준 추정은 **(i) 통과 ≈ .5**이다(아래 §7).

## 2. 단계 A에서 이 설계로 이어지는 사실 (랩 규칙: 수정은 사실에서 유도)

| 단계 A 사실 | 설계 귀결 |
|---|---|
| H_A FAIL: pooled top-2 share 0.654 < 0.70, 손실이 stem·enc2·feat·skip1 등 여러 그룹에 분산 | 사전등록 분기 → B1(표적 수정) 금지, **B2 = 전체 network 대상 교과서 QAT** |
| 세 setting(w4, a6, l50) 모두 붕괴(84/90), 세 setting의 top-2가 서로 다름(stem+skip1 / stem+enc2 / stem+feat) | 한 setting용 QAT로는 부족 → **세 압축 연산자를 모두 학습 중 노출**하는 단일 recipe |
| 진단: ch_ratio·dead_act_a6·dead_w_w4와 손실 ρ 0.25–0.50(per-tensor scale 효과), act_max는 무관 | 학습 시 fake-quant를 **record와 같은 per-tensor 규칙**으로 걸어 그 효과를 직접 최적화. 범위 클리핑(PACT) 같은 부품별 처방은 B1 영역이므로 넣지 않는다 |
| record 평가는 BN 재보정 없이 fp32 running statistics로 압축 모델을 돌린다 | 압축 batch에서는 BN을 running-stat 모드로 돌려 **평가 때와 같은 함수**를 학습 |
| **stem이 72/84 붕괴 셀에서 최대 손실 그룹**, setting별 손실 share 0.43–0.57 | 사실로만 기록(기여 2의 사실 측면). 표적 수정으로 바꾸지 않음(분기 밖). 사후 변형 C2(§8)는 별도·표기 |
| boundary heads(`shead1-3`, `gt_conv1-4`)는 inference 경로 밖 → 단독 손실 0 | 압축 학습에서도 record 연산자 집합 그대로 둠(아래 layer set) |

## 3. B2 recipe — PBE-Net compression-aware training (CAT)

### 3.1 정의 (2줄)
매 training batch마다 mode ∈ {fp32, w4_pt, a6, l50}를 균등 확률(각 1/4)로 하나 뽑아, 그 mode의 **record 연산자 그대로**(w4 per-tensor RTN, a6 per-sample dynamic per-tensor, l50 layer-wise 50 % magnitude mask)를 STE fake-op으로 걸고 protocol v1 손실을 그대로 역전파한다. 처음 10 epoch은 fp32만(warm-up), 압축 batch에서는 BN이 running statistics를 쓰고 갱신하지 않는다.

### 3.2 연산자 (평가 = record 복사본, 학습 = 같은 수식 + STE)
평가 연산자는 `A/qfix_a.py`의 `rtn_sym_tensor`, `act_hook_factory`, `prune_layer`, `Engine.apply`를 **바이트 그대로** 쓴다(9101/9601 record 복사본). 학습 연산자는 같은 수식에 STE만 더한다.

| mode | 대상 집합 (record와 동일) | 학습 forward | backward |
|---|---|---|---|
| w4_pt | 모든 floating parameter 중 `dim() ≥ 2` (PRCM `basis`, train-only heads 포함 = `Engine.targets`) | ŵ = w + (RTN4(w) − w).detach(); RTN4: scale = max\|w\|/7 (detach), round, clamp ±7 | STE identity (scale = max\|w\|라 clamp가 활성화되지 않음) |
| a6 | 모든 `_ConvNd`/`Linear` 모듈의 **출력** (`Engine.leaf`, train-only heads 포함) | ŷ = y + (Q6(y) − y).detach(); Q6: amax = per-sample max\|y\| (detach), scale = amax/31, round, clamp ±31 | STE identity |
| l50 | inference에서 실행되는 `_ConvNd`/`Linear` 중 weight `dim() ≥ 2` (`Engine.executed`: `adapter.infer`로 1장 forward hook; train-only heads 제외) | mask m = 1[\|w\| > kthvalue(\|w\|, ⌊0.5·n⌋)] (record의 "\|w\| ≤ thr → 0"과 동치), ŵ = w + (w⊙m − w).detach() | dense STE (pruned weight도 gradient를 받아 되살아날 수 있음; DPF형) |
| fp32 | — | 원 forward | — |

- l50 mask 재계산 주기 N: **l50 batch마다 매번**(현재 weight로 계산). 평가 때 규칙(현재 weight의 layer-wise 50 %)과 정확히 같은 대상이며, kthvalue 비용은 작다(§5). 학습 중엔 GPU `torch.kthvalue`, 평가 땐 record의 CPU kthvalue(값 동일, tie 규칙 동일).
- RepConv는 inference에서도 fuse되지 않는다(`net.py`의 `RepConv.forward`가 세 branch를 그대로 계산). 따라서 record는 `conv_kxk`, `conv_1x1`를 따로 양자화하고, 학습도 같은 분해 상태에서 한다 → 학습·평가 일치.
- BN(1-dim parameter)은 양자화 대상이 아니다(record와 동일).

### 3.3 학습 조건 (protocol v1 그대로)
- code snapshot `pbe_seg/code` commit e0e2f07 (source_sha256 496cc2db…), `common.trainer` 수정 없음. epochs 100, batch 16, AdamW lr 1e-3, wd 1e-2, betas (0.9, 0.999), cosine → 1e-6, grad clip 5.0, workers 4, seed = split k, DataLoader generator seed k, threshold 0.5, 입력 256, 같은 train/val/test cache.
- 손실: `seg_loss(out, side, mask) + extra_loss`(matching loss λ_s = 0.1, 보조 deep supervision) — 압축 batch에서도 같은 손실을 압축 forward에 적용.
- mode 추출 RNG: 별도 `random.Random(7919 + k)`. 전역 torch RNG(dropout)와 data order RNG를 건드리지 않는다.
- warm-up: epoch 1–10 fp32만; epoch 11–100 균등 4-way. epoch 경계는 adapter가 `infer` 호출(매 epoch val)을 보고 센다(trainer 수정 불필요).
- 압축 batch의 BN: 해당 forward 동안 모든 `BatchNorm2d`를 `eval()`(running stats 사용, 갱신 없음, affine은 학습), 끝나면 `train()` 복원. fp32 batch는 원래대로 running stats 갱신.
- **checkpoint 선택: protocol v1 그대로 fp32 val mean IoU 최대.** 근거: 사전등록 §3 "프로토콜 v1 그대로". 선택 규칙을 압축 성능으로 바꾸면 protocol 변경이 되므로 쓰지 않는다. 위험: fp32 최적 epoch가 압축 최적 epoch와 다를 수 있다. cosine 말기의 안정 구간에서 선택되므로 영향은 작다고 보지만, 이 위험은 §7에 남긴다. test는 선택에 쓰지 않는다.
- config: `model = 'pbe_cat'`, `model_cfg = DEFAULT_CFG ∪ {'cat': CAT}`, CAT = `{"modes":["fp32","w4_pt","a6","l50"],"probs":[0.25,0.25,0.25,0.25],"warmup_epochs":10,"bn_compressed":"running","ste":"identity","l50_mask_every":"each_l50_batch","mode_seed":"7919+k","op_source":"qfix_a.py b36df82f"}`. 구조는 PBENet DEFAULT_CFG 그대로(state_dict strict load 가능), config_hash는 v1과 달라진다(정상).

### 3.4 각 선택이 사전등록을 따르는 이유
- **B2 = 교과서 QAT**(§2): STE fake-quant(Bengio 2013; Jacob 2018), 학습 중 magnitude pruning mask(Zhu & Gupta 2017; DPF, Lin 2020), 여러 압축 설정을 한 모델에 학습(AdaBits/any-precision 계열)은 모두 기존 기법이다. 새 연산자, 새 손실, 부품별 처방은 없다.
- **표적 변경 없음**: 연산자·대상 집합을 record 전체 집합과 같게 두었고, stem·enc2·skip1 등 특정 부품에만 거는 것은 하나도 없다. stem 사실은 설계 입력이 아니라 기록 사항이다.
- **"보정 데이터 = train split"**(§2 B2): 학습 데이터는 train split만, val은 checkpoint 선택만, test는 마지막 1회.
- **"pruning은 양자화가 아니다"**: l50은 QAT가 아니라 pruning-aware training이다. Q 프로토콜이 세 설정을 묶어 판정하므로 recipe가 세 개를 모두 덮어야 한다. 이름은 "compression-aware training"으로 정직하게 표기한다.
- **uniform 1/4 + sampling**: 모든 mode 손실을 매 step 합하면(AdaBits식 joint) 비용이 약 4×라 16 GPU-h 예산을 넘는다(§5). per-batch random sampling은 같은 목적의 가장 단순한 확률적 근사다. 확률은 결과 전에 고정했고 튜닝하지 않는다.
- **warm-up 10 epoch, BN running mode**: from-scratch QAT의 표준 관행이다(fp32 수렴 뒤 QAT 시작; Krishnamoorthi 2018 whitepaper의 BN 통계 고정). record 평가 함수와 맞추기 위해 필요하며 결과 전에 고정했다.

### 3.5 기여 표기 (정직)
- **방법 기여 = 교과서 조합 (QAT + pruning-aware training, multi-setting sampling)** → method **중하 이하**. 새로운 방법으로 주장하지 않는다.
- **사실 기여** = (a) stem fragility: 단계 A에서 stem(encoder1 1×1 conv 3→128 + RepConv + PRCM)이 붕괴 셀 72/84에서 최대 손실 그룹이고 setting별 손실 share는 0.43–0.57이다. boundary heads는 inference 경로 밖이라 손실 0이다. (b) 라벨 없는 식별: fp32–압축 출력 일치도가 붕괴를 판별한다(v1 90셀 AUC 0.978; agree–retention Spearman 0.82/0.85/0.91).
- 단계 C 성공 시 원고 문장은 "교과서적 compression-aware training으로 PBE-Net의 압축 붕괴가 해소된다(18셀 중 n)"이다. "새 방법이 Q를 푼다"고 쓰지 않는다.

## 4. G0′ (web 2회; 기여 표기용이며 B2를 막지 않는다)

| # | query | 결과 수 | 가장 가까운 선행 |
|---|---|---|---|
| 1 | `joint quantization and pruning aware training single model robust to multiple bit-widths and sparsity (AdaBits, any-precision, DJPQ, APQ)` | link 19개(중복 제외 고유 논문 약 6편) | **DJPQ** (Wang et al., ECCV 2020, arXiv 2007.10463): pruning + mixed-precision을 한 미분 손실로 학습하지만 결과는 operating point 1개. **AdaBits** (Jin et al., CVPR 2020): 한 모델이 여러 bit-width를 지원하지만 pruning 축 없음. **APQ** (Wang et al., CVPR 2020, 2006.08509): architecture·pruning·quantization joint search이며 단일 학습 모델 아님. bit-switching 후속 연구(arXiv 2103.01435 등) |
| 2 | `quantization-aware training lightweight medical image segmentation network 4-bit robustness post-training quantization` | link 10개(고유 약 5편) | **QuantU-Net** (arXiv 2503.08719, 2025): wearable U-Net QAT(평균 4.24 bit, Brevitas). **Göksu et al. 2025**: MedSegBench 7종에서 U-Net PTQ vs QAT. **R. Zhang 박사논문(HKUST 2022)**: 의료영상 low-bit QAT/PTQ. **QMedSAM** (arXiv 2412.11186): SAM QAT |

검색 밖에서 알고 있는 관련 선행(검색 결과로 확인한 것은 아님): Any-Precision DNN (Yu et al., AAAI 2021), Robust Quantization / KURE (Shkolnik et al., NeurIPS 2020), Gradient ℓ1 regularization for quantization robustness (Alizadeh et al., ICLR 2020), CLIP-Q (Tung & Mori, CVPR 2018), Deep Compression (Han et al., ICLR 2016), DPF (Lin et al., ICLR 2020).

판정: "여러 압축 설정(bit-width·activation precision·sparsity)에 동시에 강건한 한 모델을 학습"은 선행의 직접 조합이다(AdaBits/any-precision × DJPQ/Deep Compression × DPF). 우리 recipe에서 다른 점은 연산자를 PBE-Net 평가 protocol의 record 연산자에 맞췄다는 것뿐이며, 이는 방법 novelty가 아니다. 의료 분할 QAT도 선행이 있다(QuantU-Net, Göksu 2025). → **방법 기여 = 교과서(중하 이하)로 확정 표기.** 정확 선행이 있으므로 B1이 있었더라도 이 recipe는 B2 성격이다.

## 5. 비용 추정 (단계 C1)

v1 학습 시간(PLAN.md의 dataset별 평균 train_minutes): PH2 4.1, etis 5.2, cvc 16.3, kvasir 22.3, isic17 48.6, isic18 58.2분/run → split당 154.7분. **30 run 기준 12.9 GPU-h** + run당 시작·test 약 1분(0.5 h) = 13.4 GPU-h.

압축 batch당 overhead 추정: w4 ≈ 1.03×(per-tensor RTN, functional_call), a6 ≈ 1.4–1.8×(Conv/Linear 출력마다 elementwise op 추가, memory-bound), l50 ≈ 1.05–1.15×(layer별 kthvalue + mask). warm-up 10 % 뒤 평균 ≈ (1 + 1.03 + 1.6 + 1.1)/4 ≈ 1.18 → 전체 ≈ **1.16×**(범위 1.1–1.3×; 사용자가 준 1.3–2×는 상한 쪽 시나리오).

| 항목 | GPU-h |
|---|---|
| C1 학습 30 run (1.1× / **1.16×** / 1.3× / 2×) | 14.7 / **15.4** / 17.3 / 26.3 (start·test 포함) |
| Q 평가 30 ckpt(전체 그룹 sweep 포함; 단계 A 실측 1152 s) + fp32 test 1회(학습 끝에 trainer가 계산) | 0.4 |
| timing gate(smoke, 시간만) | 0.1 |
| **합계 (중앙)** | **≈ 15.9** (범위 15.2–17.8; 2×면 26.8) |

- 2 GPU(gpu-server GPU 0·1, 3090) LPT 배분: GPU0 = isic18×5 + cvc×5 + PH2×5 (393 base min), GPU1 = isic17×5 + kvasir×5 + etis×5 (380.5 base min) → 1.16×에서 **wall ≈ 7.6 h**.
- **16 GPU-h 경계**: 중앙값이 한도와 같다. 사전등록 안에서 줄일 수 있는 것이 없다. §3이 6 ds × 5 split, protocol v1(같은 epoch)을 고정하고, 평가는 이미 작다. 그래서 **착수 전 timing gate**를 둔다. isic18 k1 train cache로 각 mode 50 step의 시간을 재고(지표 계산 0, 출력은 ms/step만), 투영 = Σ base_min × 측정 factor를 계산한다. 투영이 16 GPU-h를 넘으면 **착수하지 않고 보고**한다(사용자 결정 사항이며 사전등록 이탈). 참고로 PREREG 머리글의 총 GPU ≤20 h에서 단계 A가 0.33 h를 썼다.
- "GPU ≤16 h"를 wall-clock(2 GPU)로 읽으면 여유가 크다(≈ 8 h). 이 문서는 보수적으로 GPU-h로 읽는다.

## 6. 단계 C 구현 명세 (다음 agent용)

### 6.1 파일 (서버 `BASE=<WORK>/qfix/C1`; archive 안에는 쓰지 않음)
1. `$BASE/code/` ← `<ARCHIVE>/code` 사본(e0e2f07). `__pycache__`는 사본에서 빼고, 복사 뒤 source digest가 496cc2db…인지 확인한다(`pbe_v1_fill/queue.sh`와 같은 절차). 원본은 읽기 전용이다. 착수 전 `grep -n "def run" $BASE/code/common/trainer.py`로 서명 `run(adapter, model_cfg, dataset, k, out_dir, data_root, cache_root, device, protocol)`을 확인한다.
2. `$BASE/qfix_c_cat.py` (신규, code 밖) — adapter 객체 + runner:
   - `import models.pbe.adapter as P`. `NAME='pbe_cat'`, `DEFAULT_CFG = dict(P.DEFAULT_CFG, cat=CAT)`, `build(cfg)=P.build(cfg)`, `extra_loss=P.extra_loss`, `is_train_only=P.is_train_only`.
   - `infer(model,img)`: `STATE['saw_eval']=True; return model(img)`(fp32, val·test 평가용).
   - `forward_train(model,img)`: 첫 호출 때 대상 집합을 만든다 — targets(w4) = floating `dim≥2` params, leaf(a6) = 모든 Conv/Linear, l50 = `model.eval()` 상태로 zeros(1,3,256,256) forward hook으로 얻은 executed Conv/Linear(그 뒤 `model.train()`). `saw_eval`이면 epoch += 1. mode = fp32 if epoch ≤ 10 else `rng.choice`. fp32이면 `P.forward_train`. 압축이면 BN을 eval로 바꾸고, w4/l50은 `torch.func.functional_call(model, ste_params, (img,), {'return_aux': True})`, a6는 STE hook을 등록한 뒤 forward하고 hook을 제거한다. 반환 dict는 `P.forward_train`과 같은 키(`out, side, heads, skips=model.last_skips`). 끝에 BN을 train으로 복원한다. mode 카운트를 `STATE['mode_counts']`에 누적하고, runner가 학습 뒤 `cat_log.json`에 기록한다.
   - 연산자 함수는 `qfix_a.py`의 `rtn_sym_tensor`/`act_hook_factory`/`prune_layer` 수식을 STE 형태로 옮긴다. 평가 쪽은 import 없이 원본을 그대로 쓴다.
   - runner `--ds D --k K --gpu G`: `sys.path.insert(0,'$BASE/code')`, `STATE` 초기화(rng = Random(7919+K)), `trainer.run(adapter, adapter.DEFAULT_CFG, D, K, out_dir=$BASE/runs/pbe_cat/D/kK, data_root=…/Skin_Lesion_Segmentation/data, cache_root=…/pbe_seg/cache, device='cuda:0')`.
3. `$BASE/run_C1.sh <0|1>`: `flock -n $BASE/gpu<G>.lock`, `CUDA_VISIBLE_DEVICES=<G>`(0·1 외 거부, gpu-server-2 금지), `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`, python = `python`, `taskset`은 v1_fill과 같다. 목록은 §5의 LPT 배분을 따른다. 각 run 전 `verify_run.py pre`식 cache 존재 확인(없으면 FAIL; archive cache에 새로 쓰지 않음).
   - 명령: `cd $BASE/code && $PY -B $BASE/qfix_c_cat.py --ds D --k K 2>&1 | grep -v '^\[test\]' > $BASE/logs/cat_D_kK.log` (test 줄을 log에서 숨김; 값은 results.json에만 남는다).
4. `$BASE/qfix_c_eval.py` = `qfix_a.py` 사본. 바꾸는 곳은 `WORK`, `RUN_ROOTS=[$BASE/runs/pbe_cat]`, `assert {k:v for k,v in run['model_cfg'].items() if k!='cat'} == P.DEFAULT_CFG`, build는 `registry.get('pbe')` 그대로다(구조 동일, strict load). 연산자·지표·group 정의·sweep은 **변경 금지**. 출력 `$BASE/eval/out/<ds>_k<k>.json`과 `C1_attribution.json`(단계 A와 같은 형식이라 stem 사실의 전후 비교 가능).
5. `$BASE/qfix_c_decide.py` (torch 불필요): §6.4의 판정. 입력은 `C1_attribution.json`, `B_baseline_18cells.json`, v1 30 run과 C1 30 run의 `results.json`(test dice)이다.
6. 단계 C 시작 시 각 파일의 sha256을 `C_PLAN.md`에 적고 고정한다.

### 6.2 표식·재시작 안전
- run 단위: `DONE`(results.json status complete, history 100행, config.model == 'pbe_cat', model_cfg.cat == CAT, data_cache 이름 == v1 같은 (D,k)), `FAIL`(비정상 종료 또는 `FAILED.txt`; trainer가 예외를 삼키므로 반드시 검사), `MISMATCH`(검증 실패, 결과는 보존). `$BASE/ALL_DONE`에 DONE/FAIL/MISMATCH 수를 적는다.
- 재시작: DONE이면 건너뛴다. trainer는 run 중간 재개가 없으므로 미완 run은 처음부터 다시 돈다(seed 고정이라 같은 결과). 기존 결과·checkpoint는 지우지 않는다(덮어쓸 상황이면 `_attempt<n>` 폴더로 보존).
- 평가: `qfix_c_eval.py --run`은 단계 A처럼 `<tag>.DONE`이면 건너뛰고, 모든 cell 뒤 weight byte-exact 복원을 확인한다.
- isic17 k2는 과거 CUDA backward error 이력이 있다. 실패하면 GPU를 의심하고 1회 재시도한다(같은 설정).

### 6.3 test 봉인
- test label을 쓰는 계산은 trainer가 각 run 끝에 선택된 checkpoint로 1회 하는 `evaluate(te)`뿐이다. 값은 results.json에만 있고 log에서는 숨긴다. 30 run과 Q 평가가 모두 DONE이 된 뒤 `qfix_c_decide.py`가 **한 번** 읽는다. 그 전에는 누구도 test 필드를 열지 않는다.
- 라벨 없는 일치도는 단계 A와 같다(test 앞 8장의 image만, masks 미로딩). 기준 (iii) 계산에만 쓴다.
- 결과를 본 뒤 recipe·선택 규칙·임계값 변경(튜닝) 금지(§5 운영).

### 6.4 판정 계산 (사전 고정)
기호: d ∈ 6 datasets, s ∈ {w4_pt, a6, l50}, k ∈ 1..5. R(d,s,k) = Dice_val(q) / Dice_val(fp32), 같은 C1 checkpoint, `qfix_c_eval`의 "all" cell.
- **(i)** N = Σ_{d,s} 1[ (1/5) Σ_k R(d,s,k) ≥ 0.8 ]. **통과 ⇔ N ≥ 16** (/18). 보조 보고: checkpoint 단위 Σ 1[R ≥ 0.8] / 90, setting별 N.
- **(ii)** D^X_test(d,k) = results.json `test.dice`(X ∈ {v1, C1}). Δ_d = mean_k D^C1 − mean_k D^v1, σ_d = 표본 SD_k(D^v1(d,·)) (n−1). **통과 ⇔ 모든 d에서 Δ_d ≥ −σ_d** ("v1 대비 split SD 안" = 하락 폭이 v1 split SD 이하; 상승은 실패가 아님). 보조로 양측 |Δ_d| ≤ σ_d와 val Dice 기준도 보고한다. v1 = 단계 A의 30 run(archive 20 + pbe_v1_fill 10).
- **(iii)** C1 90 cell(run × s)에서 score = 1 − agree_all, label = 1[R < 0.8], AUC = Mann–Whitney U/(n₊n₋)(tie 0.5). 붕괴·건강 중 한쪽이 5개 미만이면 사전 고정 대체로 v1 90 + C1 90 = 180 cell pooled AUC를 쓴다(v1 단독 0.978). **통과 ⇔ AUC ≥ 0.95.**
- 결론 매핑(PREREG §3 그대로): (i)(ii)(iii) 전부 → "PBE-Net이 Q에 답한다"(기여 2 = 수정된 PBE-Net, 방법 = 교과서 CAT로 표기). (i)만 실패 → 부분(N/18 보고, "효율 + 부분 강건"). (ii) 실패 → 정확도–강건성 trade-off로 보고하고 주장하지 않는다. (iii)만 실패하는 경우는 사전등록에 매핑이 없으므로 "채택 아님"으로 처리하고, 라벨 없는 식별 주장을 빼고 보고한다(새 규칙을 만들지 않음).

## 7. 한계·위험 (사전 기록)
1. **단계 A 합성 역검증 FAIL**(frozen rule): 심은 enc3 outlier가 top-1이 되지 못했다(w4 rank 2, a6 rank 3). 사후로 보면 변화는 enc3에만 나타나 귀속 자체는 부품 특이적이다. 그러나 native stem 손실보다 작은 효과는 top-1로 잡지 못한다 → **귀속 방법의 검출력이 약하다**. stem fact는 "가장 큰 단독 손실"로만 주장하고, 다른 부품의 기여를 배제하는 근거로 쓰지 않는다.
2. 기준선이 0/18이라 (i)은 +16셀 개선을 요구한다. l50(평균 0.049)이 가장 어렵다. 추정: (i) 통과 ≈ .5, (ii) 통과 ≈ .7(압축 batch 75 %가 fp32 정확도를 깎을 위험), (iii) 통과 ≈ .6(건강 셀이 많아지면 붕괴 표본이 줄어 AUC가 불안정하고, 대체 규칙으로 감).
3. checkpoint 선택이 fp32 val IoU라 압축 최적 epoch와 어긋날 수 있다(§3.3).
4. 비용 중앙값이 16 GPU-h 경계에 있다 → timing gate(§5).
5. G0′ 2회는 2023년 이후 선행을 충분히 덮지 못했을 수 있다. 다만 방법 기여를 이미 최저로 표기하므로 결론은 바뀌지 않는다.

## 8. C2 (post hoc, marked) — stem 표적 변형 1종, 명세만, 실행하지 않음
**표기: 사전등록 분기 밖(H_A FAIL → B1 금지).** C1 결과 뒤 study lead 요청으로 추가한 사후 탐색이다. 원고에서는 "post hoc"으로 분리하고 기준 (i)–(iii) 판정에 넣지 않는다.

- 유도: stem fact만 쓴다(72/84 top-1). 진단(ch_ratio, dead_w_w4, dead_act_a6 ρ 0.25–0.50)은 per-tensor scale 하나가 채널 간 큰 범위 차이를 덮지 못한다는 점을 가리킨다. stem의 1×1 conv(3→128, weight 384개)는 출력 채널 scale이 자유 gauge다. 바로 뒤가 RepConv이고, 그 세 branch가 모두 BN(`bn_kxk`, `bn_1x1`, `bn_identity`)에 들어가기 때문이다.
- **선택 변형(1종, 사전 고정): stem cross-layer equalization (DFQ/CLE, Nagel et al. 2019를 stem 한 쌍에만 적용).** 1×1 conv 출력 채널 c에 s_c를 곱하고, 같은 채널의 `conv_kxk[c]`, `conv_1x1[c]` weight를 1/s_c 배 한다. 1×1 conv에 bias가 있으면 bias도 s_c 배 한다. `bn_identity`의 running_mean·running_var는 각각 s_c, s_c² 배 한다. 그러면 fp32 함수가 보존된다(eval BN). 단 `bn_identity`의 eps 항은 eps/s_c²로 바뀐 것과 같아 엄밀히는 근사다. 단계 A 합성 검사(×1000)에서도 Dice가 변하지 않았다. s_c = sqrt(r₂_c / r₁_c)이며 r₁_c = max\|W₁[c]\|, r₂_c = max\|W₂[c]\|이다(W₂ = conv_kxk와 conv_1x1 중 큰 쪽). 단일 pass, 하이퍼파라미터 0. **record 연산자(per-tensor RTN, a6, l50)는 바꾸지 않는다.** 반면 per-channel scale 안(案)은 평가 quantizer를 바꾸므로 Q 프로토콜과 달라져 채택하지 않는다. RepConv fuse 후 양자화 안은 평가 대상 그래프를 바꾸므로 보조 기록만 한다.
- 검증(실행 시): fp32 val Dice 변화 |Δ| < 1e-4(함수 보존 확인. 단계 A 합성 검사와 같은 BN 보정 논리). 선택할 하이퍼파라미터가 없으므로 val-only 선택 조건을 자동으로 만족한다. 각 split 1회 적용한다.
- 대상과 비용: 학습 없음. C1 30 ckpt(주)와 v1 30 ckpt(참고)에 적용한 뒤 `qfix_c_eval.py` "all"만 평가 → ≈ 0.1–0.4 GPU-h. **총 GPU(A 0.33 + C1 ≈ 15.9 + C2 ≤ 0.4) ≈ 16.6 h ≤ 20 h.**
- 보고: (i)식 18셀 수와 setting별 평균 R을 C1 대비 나란히 적고, "post hoc, 분기 밖, 선행 = DFQ"로 표기한다. 방법 기여는 C1과 같이 교과서다.
