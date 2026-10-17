# F5 pre-registration — PBE-Net under the FIVES T protocol

**FROZEN 2026-10-16 ([internal decision] decisions below; draft kept as PREREG_F5_draft_v0.md).**
작성 2026-10-16 (journal review 1 §6 결정 F5). 측정·학습·launch 없음. 이 문서 작성 시점에 F5용 학습/추론은 한 번도 실행되지 않았음. 사용한 정보: G2 결과(공개된 3 모델 수치), G2 코드, PBE-Net 코드(read-only), gpu-server read-only listing.

Freeze 조건: 코디네이터 확인 → 아래 §0 open choices 결정 반영 → 이 파일의 sha256을 §7 표에 기록 → 그 뒤 synthetic reverse-check(§6) → launch. freeze 이후 변경은 하단 Amendments에만 기록.

---

## 0. Open choices (코디네이터 결정 필요; 괄호 안이 이 draft의 default)

| # | 항목 | 선택지 | default와 근거 |
|---|---|---|---|
| O1 | PBE-Net loss | (a) protocol v1 loss 유지: BCE+soft Dice(main) + 5 side outputs(0.1..0.5) + train-only matching loss λ_s=0.1 / (b) main BCE+Dice만 | **(a)**. G2 학습 코드(`g2a_train.py` l.210)는 원래 `seg_loss(out, side) + adapter.extra_loss(...)`를 계산하며, G2의 "BCE+Dice"도 모델 고유 deep supervision을 포함(UCMNet 5 side outputs가 그렇게 학습됨). 따라서 (a)는 G2 recipe를 **코드 수정 없이** 그대로 적용한 결과이고, Table 1 PBE-Net과 같은 모델 정의. (b)는 코드 수정이 필요하고 PBE-Net의 정의(학습 전용 head)를 바꿈 |
| O2 | T 진술의 판정 기준 | (a) point ratio ≥ 0.8 **그리고** bootstrap 95% CI 하한 ≥ 0.8 / (b) point ratio ≥ 0.8만(G2 B6 연속성) | **(a)**. 심사자 R3가 CI 없는 점추정을 지적. (b)는 G2 B6과 같은 기준이므로 "point" 판정은 표에 함께 보고 |
| O3 | AMP 행을 T 진술에 포함 | 포함 / 기술(descriptive)만 | **기술만**. MAResUNet AMP가 100/100 non-finite였던 전례; AMP는 별도 행과 non-finite 수로 보고 |
| O4 | patch 512 행을 T 진술에 포함 | 포함 / robustness 줄로만 | **포함**(0.5 GiB 실행 가능할 때만). G2 B3 한계(256 crop 가중치로 512 추론)는 문장으로 명시. PBE-Net은 512 tile이 0.5 GiB를 넘을 가능성이 큼(§8 P2) |
| O5 | GPU-h cap 초과 시 drop 순서 | rs1024 → rs512 → rs256 / 다른 순서 | **rs1024 → rs512 → rs256**(native는 drop 불가, T 진술의 근거) |
| O6 | 학습 OOM(batch 16) 시 | 정지·보고 / gradient accumulation 2×8 허용 | **정지·보고**(amendment 없이 recipe 변경 금지) |
| O7 | seam 비교 grid | G2 정의(각 variant 자기 grid = stride) / 공통 grid(=patch) | **G2 정의 primary**, 공통 grid p는 descriptive로 함께 출력 |
| O8 | 0.5 GiB 경계의 Resize 행 | Resize 256/512/1024 모두 표에 / 1024는 부록 | **모두 본 표에**(F5 요구: Resize 1024 행) |

### 0.1 study lead decisions ([id], 2026-10-16) — frozen; these override the defaults above where they differ
- O1 (a) protocol v1 loss as is (Table 1 PBE-Net의 같은 학습이 논지).
- O2 (a) CI lower bound. indeterminate는 그대로 보고. UCMNet p256 o0가 indeterminate이면 4f 문장도 "indeterminate"로 맞춤(사실 문장 변경은 결과 뒤 코디네이터에게).
- O3 AMP는 T 판정에서 제외, 보조 행으로만(fp32 배포 전제). non-finite 행은 기록.
- O4 p512는 peak ≤ 512 MiB이면 자동 포함(규칙 E가 결정; 사전 예측 0.35 기록).
- O5 drop 순서 rs1024 → rs512 → rs256.
- O6 학습 OOM → 정지·보고. gradient accumulation 불가(학습 레시피 변경).
- O7 seam 격자 = G2와 동일 격자(각 variant 자기 grid = stride). 공통 grid descriptive 출력은 하지 않음.
- O8 Resize 행은 같은 표의 별도 블록 "resize, not tiled"; T 판정 대상 아님(overlap 귀속과 무관).
- PRCM 전역 평균 gate가 tiling에서 맥락을 바꾼다는 점은 §8 P3에 예측으로 기록됨. 결과 뒤 해석 문장은 그 예측 적중 여부로만 쓴다.

---

## 1. 학습 (G2 Part A/B recipe 그대로; PBE-Net 4 trainings)

- 모델: registry `pbe`, `DEFAULT_CFG` = `{'c_list': [128,192,256,384,512], 'skip':'predictive', 'block':'ebpb', 'match':'cos', 'lambda_s':0.1, 'band_kernel':3}` (band_stride 기본 4). code_v2에서 import, 수정 없음.
- Variants: `native`(2048 image의 uniform random 256×256 crop, label native), `rs256`(image·label area-downscale ×8, whole 256 image), `rs512`/`rs1024`(×4/×2 downscale 후 random 256 crop). label은 downscale 후 threshold 0.5. inference: rs는 downscaled whole image → probability bilinear upsample to 2048 → threshold 0.5.
- 공통: FIVES official TRAIN 600장 중 `g2a/val_list.txt`의 100장 = train-internal validation, 나머지 500장 = 학습. **official FIVES test folder는 열지 않음**(G2 코드는 train/ 경로만 읽음). per-image min-max [0,1](2048 전체 기준). AdamW lr 1e-3, wd 1e-2, betas (0.9, 0.999), cosine to 1e-6 over 8000 steps, grad-norm clip 5.0(non-finite norm이면 update skip·count), **batch 16, 8000 steps, seed 1**, aug = hflip, vflip, rot90(독립), cudnn deterministic.
- Loss(O1 default (a)): `seg_loss(out, side, y)` = BCE+soft Dice(main) + Σ w_i·BCE+Dice(side_i), w = 0.1,0.2,0.3,0.4,0.5(PBE-Net은 gt_conv1–4, side[0] 중복 = legacy GT tuple) + `extra_loss` = 0.1 × centred-cosine matching loss on 3×3 morphological-gradient bands at skip heads `shead1–3`. 학습 전용 head(`base.gt_conv*`, `shead*`)는 inference(`adapter.infer` = main output만)에 쓰이지 않음. native 학습에서 band는 native-resolution 256 crop의 GT에서 계산(rs는 downscaled label).
- Checkpoint: 1000 step마다 100 val 이미지 mean per-image Dice at own resolution(native: 2048 via non-overlapping 256 tiles; rs R: R×R whole image vs thresholded downscaled label), best = `best.pth`(tie → earliest), `final.pth`도 보관. S 평가는 `best.pth`.
- 코드: `scripts/f5_train.py` = `scripts/g2a_train.py`의 **byte-identical copy**(sha256 910fb8608f9c1c55…; `--model`에 choices 제한 없음, registry에 `pbe` 등록됨 → 수정 불필요). 출력 `<WORK>/s4mem/f5/pbe/{native,rs256,rs512,rs1024}/`. cache는 `f5/cache` → `../g2a/cache` symlink(read-only 사용, 복사·재생성 없음; `PREP_DONE` 존재 확인됨).
- Launch(사용자 또는 확인 후): GPU 0만, `CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`, 순서 native → rs256 → rs512 → rs1024. GPU 1과 gpu-server-2은 사용하지 않음. 종료는 PID로만.

## 2. S set·peak memory·budget (G2 B3/B4 그대로)

PBE-Net variants(총 14행): `rs256_fp32`, `rs512_fp32`, `rs1024_fp32`; native에서 `tile_p{256,512}_o{0,50}_{fp32,amp}`(8행, Gaussian blending σ=patch/8, max 1 정규화, probability 평균, threshold 0.5; stride = patch×(1−overlap)); `full_fp32`, `full_amp`. full_fp32 GPU OOM이면 CPU fp32로 계산하고 모든 budget에서 non-executable 표시. OOM-boundary probe 256..2048 step 256, fp32/AMP, batch 1.

Peak memory = `torch.cuda.max_memory_allocated`, validation image 0의 batch-1 inference 1회(tile batch 1, fp32 weights 상주, 2048² probability·weight accumulator GPU 상주, input은 CPU에서 tile 단위 복사; resize/full은 input 전체 GPU). CUDA context 제외, reserved는 기술용. **Executable at B iff peak ≤ B, B ∈ {2 GiB = 2048 MiB, 0.5 GiB = 512 MiB}.** metric pass는 tile batch 8(p256)/4(p512), 동일 산술. 원고의 "R=512 whole-image 0.525 GiB"는 다른 측정이므로 F5 표는 이 정의의 값만 쓴다.

## 3. Metrics와 bootstrap

- Per image(G1 `g1.py` `prep_gt/counts/fin` 그대로, G2 `eval_task`): Dice, clDice(skimage skeleton), **thin-vessel clDice k=3 (primary)**, k=5(기술), seam band/interior clDice(grid stride 128/256/512, band = seam line ±8 px, skeleton은 전체 이미지에서 계산). 
- Median: 100 val 이미지. thin3는 GT thin skeleton이 있는 이미지 집합 I3(모든 variant·모델 공통, `thin3_nskel` > 0)에서.
- **Ratio** r(v) = median thin3(v) / median thin3(full_fp32, 같은 모델).
- **Seam relative** s(v) = [median band(v)/median interior(v)] / [median band(full_fp32)/median interior(full_fp32)], 같은 grid(O7). seam break flag iff s(v) < 0.8(기술).
- **Bootstrap**: paired percentile bootstrap, **B = 2000**, `numpy.random.default_rng(20261016)`. 이미지 인덱스 행렬을 metric 집합별로 한 번 생성(전체 100장용 1개, I3용 1개; 생성 순서 = all → I3)하고 모든 모델·variant에 공유(같은 replicate = 같은 이미지 multiset → variant 간·모델 간 paired). 각 replicate에서 median, ratio(분자·분모 같은 resample), seam relative, overlap contrast를 다시 계산; 95% CI = 2.5/97.5 percentile. **보고하는 모든 median과 ratio에 CI를 붙임.** non-finite 이미지가 있는 AMP variant는 "non-finite k/100"으로 표기하고 ratio·CI를 계산하지 않음(G2의 0.000 표기 대신).
- Overlap contrasts(patch p별): Δseam(p) = s(tile_p_o50) − s(tile_p_o0), Δratio(p) = r(tile_p_o50) − r(tile_p_o0), 둘 다 paired bootstrap CI.

## 4. 보고 규칙 (mechanical; `f5_summary.py`가 문자열 생성)

**4.1 T 진술 (PBE-Net, primary).** E = PBE-Net fp32 tiled variants 중 peak ≤ 512 MiB인 집합(O4 default: p256·p512 모두 후보). 각 v ∈ E 분류:
- **preserved**: r(v) ≥ 0.8 **그리고** CI 하한 ≥ 0.8 (O2 (a));
- **not preserved**: CI 상한 < 0.8;
- **indeterminate**: 그 외(point ≥ 0.8이나 하한 < 0.8, 또는 point < 0.8이나 상한 ≥ 0.8).
모델 단위 진술(정확한 문구 틀):
- E가 비면 → "PBE-Net: no tiled fp32 variant fits 0.5 GiB (VACUOUS)".
- E 전부 preserved → "Tile inference of PBE-Net within 0.5 GiB keeps thin-vessel clDice at ≥ 0.8 of full-resolution inference (all executable tiled variants, CI lower bounds ≥ 0.8)."
- 일부 → "… preserves at {preserved 목록}, except {not preserved / indeterminate 목록과 r [CI]}."
- 없음 → "Tile inference of PBE-Net within 0.5 GiB does not keep thin-vessel clDice at 0.8 of full resolution."
point-only 판정(O2 (b), G2 B6 기준)도 같은 표에 열로 함께 출력. 절대값(thin3 clDice, clDice, Dice, peak)을 ratio보다 먼저 보고.

**4.2 같은 규칙의 G2 3 모델 재계산(재학습·재추론 없음).** 입력 = `g2a/{ucmnet,utanet,maresunet}/{native,rs256,rs512,rs1024}/results.json`의 `per_image`(read-only, CPU). 4.1과 같은 bootstrap 행렬·분류·문구로 모델별 T 진술 생성, 원고 Table `tab:tiling_fives`의 point 값과 일치하는지 assert(불일치 → 버그로 기록, 결과 해석 중단). 원고 현 진술 "tile inference within 0.5 GiB preserves thin-vessel topology, with UCMNet at 256-pixel patches as the one exception"은 이 재계산 결과로 대체/확인한다(UCMNet tile_p256_o0_fp32 r = 0.790은 indeterminate가 될 수 있음 — 사전 명시).

**4.3 Overlap 귀속.** overlap 효과는 **seam relative에만** 귀속:
- "overlap raises seam continuity" iff Δseam(p) CI 하한 > 0; 상한 < 0이면 "lowers"; 그 외 "no detectable seam effect".
- Δratio(p)는 표에 수치·CI만 보고하고 원인 귀속 문장을 쓰지 않음. 허용 문장: CI가 0 포함 → "whole-image ratio does not differ detectably between 0% and 50% overlap"; CI가 0 미포함 → 방향만 기술 + "not attributed to overlap"(0% > 50% 패턴의 untested candidate = crop/input size mismatch).
- T 진술(4.1/4.2)에 overlap을 근거로 쓰지 않음.

**4.4 기타.** Resize 256/512/1024, full_fp32/full_amp 행은 모두 표에. AMP 행은 descriptive(O3). patch-512 행은 "weights trained on 256 crops" 주석. 1 dataset·1 seed·100 train-internal val·selection과 같은 val 사용(optimism, 모든 variant 동일)을 limitation으로 명시. official test 미사용.

## 5. Runtime 추정과 stop rules (gpu-server GPU 0, RTX 3090 24 GiB)

근거(read-only listing): G2 queue는 12 trainings + 12 evals에 ~3.5 h wall(FIRST_DONE 13:23 → ALL_DONE 16:05 UTC, Part A ~1 h 포함). 학습 시간 UCMNet 15 min, UTANet 22–25 min, MAResUNet 10–12 min; native eval(10 variants × 100 images + probe) ≈ 6–10 min(utanet `[eval] 91/100 308s`).

PBE-Net 추정: c0 = 128 channel이 crop 해상도 전체에서 동작 → step당 UTANet 이상으로 가정. 학습 4 × 20–45 min = 1.3–3.0 h; native eval 0.2–0.6 h(full_fp32 2048 GPU 가능 가정; CPU fallback이면 CPU 시간이 늘지만 GPU-h 아님); rs eval 3 × 3–6 min; smoke 0.1 h. **합계 ≈ 1.7–3.9 GPU-h, point ≈ 2.7 GPU-h. Cap 6 GPU-h.** G2 3 모델 bootstrap 재계산과 synthetic check는 CPU only(< 0.5 CPU-h).

Stop rules:
- S0: launch 직전 GPU 0에 다른 process가 있으면 launch하지 않고 보고.
- S1: smoke(`--steps 50 --n-train-limit 32`, 별도 out `f5_smoke/`, cache symlink) 통과 조건 = finite loss, `best.pth` 생성, eval까지 완료. 실패 → 정지·보고(코드 수정은 amendment).
- S2: native 200 step 후 s/step으로 4 trainings + eval 총시간을 투영해 > 6 GPU-h이면 O5 순서로 variant를 drop(native 불가), drop 목록을 summary에 기록.
- S3: non-finite loss → 해당 run FAIL(재시도·설정 변경 없음). skipped updates > 80(1%)이면 flag.
- S4: 학습 OOM → 정지·보고(O6).
- S5: 첫 launch부터 wall 6 h에 미완 run은 PID로 kill, summary는 `Jobs complete: NO`, 판정은 provisional.
- 실패·누락 run을 결과를 본 뒤 재실행하지 않음(코드 버그만 amendment 후 재실행, 이전 출력은 이름 변경 보관, 삭제 금지).

## 6. Synthetic reverse-check (CPU, 실데이터 분석 전 1회; 전부 통과해야 §3–4 실행)

`f5_boot.py`·`f5_summary.py`를 실데이터와 같은 함수로 synthetic 입력에 적용(seam metric은 G2 `_band`/`_region_cl`·`g1.py` import, 수정 없음):
- R1 ratio/bootstrap: 100 images, full thin3 ~ Beta(5,5), variant = c·full × lognormal(0, 0.1) noise, c ∈ {0.70, 0.80, 1.00}. 통과: 200 simulated datasets에서 95% CI의 true ratio-of-medians coverage ∈ [0.90, 0.99]; c=0.70 → "not preserved" ≥ 95%; c=1.00 → "preserved" ≥ 95%; c=0.80 → "indeterminate"가 최빈.
- R2 seam: 2048² synthetic vessel map(폭 1–5 px random polylines), GT = map. (A) pred = GT → s ∈ [0.98, 1.02]; (B) stride 256 seam line을 가로지르는 vessel에 4 px gap → band clDice 하락, interior 변화 ≤ 0.01, s < 0.8, flag on; (C) 같은 수의 gap을 모든 seam에서 ≥ 16 px 떨어진 위치에 → s ≥ 0.9(interior 결손을 seam으로 오귀속하지 않음).
- R3 overlap 귀속: synthetic o0/o50 쌍에서 whole-image ratio 동일·seam만 다름 → 규칙 출력이 "raises seam continuity" + "does not differ detectably"; 반대로 seam 동일·ratio 다름 → "no detectable seam effect" + 방향 기술 + "not attributed to overlap".
- R4 pairing: variant = full(동일 per-image 값) → ratio CI = [1, 1] 정확히.
실패 시 실데이터 분석 금지, 수정은 amendment, synthetic 재실행.

## 7. sha rule·result once

Freeze 시 아래 sha256을 기록(gpu-server에서 `sha256sum`; 현재 확인 값은 prefix):

| file | sha256 (prefix, 2026-10-16 확인) |
|---|---|
| `scripts/f5_train.py` (= g2a_train.py) | 910fb8608f9c1c55 (copy 후 동일해야 함) |
| `scripts/g1.py` | 1625914b9c55d2db |
| `scripts/g2a_summary.py` (참조) | 8b91ae05a77f4de9 |
| `code_v2/models/pbe/src/net.py` | de4dcb9c59d4a49c (local 동일) |
| `code_v2/models/pbe/src/matching.py` | 3af81e50e47e4499 (local 동일) |
| `code_v2/models/pbe/adapter.py` | 23e061bbb1a88cf6 (local 동일) |
| `code_v2/common/losses.py` | 365a663073c303f8 (local 동일) |
| `g2a/val_list.txt` / `g2a/split.json` | 93ddae6ab100eb37 / 840d22ff92ff5339 |
| `scripts/f5_boot.py`, `f5_summary.py`, `f5_queue.sh` | freeze 시 기록(신규) |
| `PREREG_F5.md` (이 파일, frozen 판) | freeze 시 기록 |

참고: pbe `manifest.json`의 net.py sha(8b78f92e…)는 현재 파일과 다름(이후 `last_skips` 노출 추가로 보임; forward 출력 불변). F5는 현재 파일 sha를 기준으로 함.

규칙: summary 생성 시 위 파일의 sha를 다시 계산해 출력에 기록, 하나라도 다르면 결과는 "invalid until amendment". **Result once**: 학습·평가·bootstrap·summary는 frozen 설정으로 한 번 실행하고 그 결과를 보고; threshold(0.8), k, patch, overlap, budget, B, seed, step 수를 결과를 본 뒤 바꾸지 않음. 버그 수정 재실행은 amendment로만, 이전 산출물은 suffix로 보관(삭제 금지).

## 8. 사전 예측 (descriptive, gate 아님)

- P1 PBE-Net native best val Dice ≥ 0.80 (p = 0.8).
- P2 tile_p256 fp32 peak ≤ 512 MiB (p = 0.9); tile_p512 fp32 ≤ 512 MiB (p = 0.35); full_fp32 > 2 GiB (p = 0.9); rs512 > 512 MiB (p = 0.6), rs1024 > 512 MiB (p = 0.95).
- P3 PBE-Net tile_p256_o50_fp32 r ≥ 0.8 with CI 하한 ≥ 0.8 (p = 0.6). 근거: PRCM channel gate가 전역 평균(x.mean over H,W)을 쓰므로 tile마다 gate 문맥이 달라짐(MAResUNet global attention과 유사하나 channel-only); 경량 UCMNet은 p256에서 실패.
- P4 Δseam(256) CI 하한 > 0 (p = 0.8); Δratio(256) CI가 0 포함 또는 음수 (p = 0.75).
- P5 rs256 r < 0.8 (p = 0.85).
- P6 PBE-Net AMP에서 non-finite 이미지 ≥ 1 (p = 0.2).
- P7 G2 재계산: UCMNet p256 두 행 외 모든 fp32 tiled 행이 preserved (p = 0.85); UCMNet tile_p256_o0_fp32(0.790)는 indeterminate (p = 0.6).

## 9. 산출물

서버 `<WORK>/s4mem/f5/`: `pbe/<variant>/{best.pth,final.pth,train_info.json,results.json}`, `logs/`, `markers/`, `F5_SYNTH.md`(§6), `F5_BOOT.json`(PBE-Net + G2 3 모델 replicate 요약), `F5_SUMMARY.md`(표·T 진술·overlap 문장·sha·Jobs complete). 로컬 사본은 이 폴더에 `F5_SUMMARY_server_copy.md`.

## Amendments
(none yet. AMENDMENT 0 will record sha of f5_queue.sh, f5_boot.py, f5_summary.py and the synthetic-check result before any results.json is read.)
