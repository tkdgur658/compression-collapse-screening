# PREREG_ABL — F4: PBE-Net 구성요소 one-factor ablation (protocol v1)

**FROZEN 2026-10-16 15:37 KST (06:37 UTC).** 코디네이터 결정(18:3x 지시: 6 plain block, 7 SE, 8 datasets)을 반영해 동결했다. 이 파일의 sha256은 `prereg_release/SHA256SUMS.txt`(06_PREREG_ABL.md)와 서버 `BASE/SHA256SUMS`(PREREG_ABL.md)에 기록한다. 이후 수정은 말미 AMENDMENT로만 한다. 동결 전 실측: FLOPs(thop, 로컬 CPU), 기본 경로 bitwise 동일성·smoke·timing(GPU 1, 학습 지표 출력 없음), 합성 역검사(기존 결과 파일만 읽음). F4는 F3(GPU 0)와 병행해 GPU 1에서 launch한다(코디네이터 결정, A7). cap은 GPU 1에서 60 GPU-h다. 동결 전 DRAFT는 `abl/work/PREREG_ABL_DRAFT_pre_freeze.md`에 보존했다.

근거 문서: `journal_review_1/SUMMARY.md` §3 F4·§6, REVIEWER_1 M4·E1(4 변형 × 3 ds × 5 split 제안), REVIEWER_2 M8·E5, `manuscript_cleanup/text/3_method/3b_model.tex`, `code/pbe_seg/models/pbe/{adapter.py, src/net.py, src/matching.py}`(로컬)와 서버 학습 코드 `<ARCHIVE>/code`(commit e0e2f07; adapter.py sha 2ffa5b7f…e3d5, net.py 8b78f92e…9668, matching.py 2f316e28…0066), `configs/rows.json`, `manuscript_cleanup/TABLE1_SIG.md`, `manuscript_cleanup/pbe_v1_fill/PLAN.md`.

---

## 0. 기준 (변경 없음)
- **full PBE-Net = 기존 v1 30 runs**: archive `<ARCHIVE>/runs/main/pbe/<ds>/k<k>`(20) + `<WORK>/pbe_v1_fill/runs/pbe/<ds>/k<k>`(10). 비교에는 선택된 3 ds × 5 split = 15 run만 쓴다. 새로 학습하지 않는다.
- **protocol v1 그대로**(`pbe_v1_fill/PLAN.md` 표): epochs 100, batch 16, AdamW lr 1e-3, wd 1e-2, betas (0.9, 0.999), cosine → 1e-6, grad clip 5.0, workers 4, checkpoint = `max_val_mean_iou`, threshold 0.5, input 256. **seed = split index k**(`trainer.py:54` `seed_everything(k)`, `:57` DataLoader generator). 같은 원 캐시 `<ARCHIVE>/cache/v1/<ds>/{train,val,test}_<k>_<key>.npz`(읽기만).
- 코드: e0e2f07 사본(`pbe_seg/code`, source digest 496cc2db…bf7). full v1 runs와 같은 코드다. 변형 4개는 이 사본에 아래 §2 변경만 더한 private snapshot에서 학습한다.

## 1. 데이터셋 선택 (결과 전 고정)
1. **CVC-ClinicDB(`cvc`)**: Table 1에서 유의한 데이터셋이다(MK-UNet 대비 +0.023, Holm p = 0.0225, `TABLE1_SIG.md`).
2. **가장 큰 데이터셋 = ISIC 2018(`isic18`)**: 기준은 train 영상 수다. train_1 캐시 크기는 isic18 424,201,518 B, isic17 338,249,070, kvasir 209,808,750, cvc 128,465,030, etis 41,423,926, PH2 31,462,830 B이다. 영상 1장(3×256×256 + 1×256×256 uint8)은 262,144 B이므로 약 1618 / 1290 / 800 / 490 / 158 / 120장이다. v1 train_minutes 평균(isic18 57.1 > isic17 43.4)과도 순위가 맞는다. **확인(동결 전, k1 캐시 `images.npy` header)**: train/val/test = isic18 1618/538/538(전체 2694), isic17 1290/430/430(2150), kvasir 800/100/100, cvc 490/61/61, etis 158/19/19, PH2 120/40/40. train 수와 전체 수 모두 isic18이 가장 크다(A2 해결).
3. **무작위 1개(ETIS-Larib 제외)**. 고정 순서 = Table 1 열 순서 `['PH2','isic17','isic18','kvasir','cvc','etis']`에서 cvc, isic18, etis를 뺀다:
```python
import numpy as np
order = ['PH2', 'isic17', 'isic18', 'kvasir', 'cvc', 'etis']
rem = [d for d in order if d not in ('cvc', 'isic18', 'etis')]   # ['PH2', 'isic17', 'kvasir']
rs = np.random.RandomState(7919)
i = rs.choice(len(rem))
print(rem, i, rem[i])
```
출력(2026-10-16, numpy 2.2.4 로컬; RandomState는 legacy MT19937이라 버전에 따라 바뀌지 않음): `['PH2', 'isic17', 'kvasir'] 1 isic17`. 같은 seed로 `rs.choice(rem)`을 써도 `isic17`이다.

**선택 = {cvc, isic18, isic17} (CVC-ClinicDB, ISIC 2018 = largest, ISIC 2017 = seed 7919 draw). 코디네이터 결정 8.** 기록: "two ISIC datasets included by the fixed rule". 두 ISIC 데이터셋이 같은 피부 병변 영역이고, isic17에서 PBE-Net은 best baseline보다 −0.002다. 이 선택은 고정 규칙의 결과이며 바꾸지 않는다(A3 해결). 원고에도 같은 문장으로 적는다.

## 2. 변형 4개 (one-factor; 3b_model.tex·net.py 구성요소 이름 기준)
줄 번호는 서버 e0e2f07 net.py(로컬 `code/pbe_seg/models/pbe/src/net.py`도 같은 줄)와 `3b_model.tex`다.

| ID | 변형 | 원고 구성요소 | 코드 위치 | config key | 코드 변경 |
|---|---|---|---|---|---|
| **V1 idskip** | predictive-residual skip s_i = e_i − U(W_i e_{i+1}) (i = 1..3)를 plain skip s_i = e_i로 바꾼다. s_4 = e_4는 그대로 둔다 | 3b:30–47 "Predictive-residual skip", Eq. `eq:skip`(3b:35), 3b:38, 3b:43 | net.py:163-167 (`if self.skip == 'predictive'` … `else: s1, s2, s3 = e1, e2, e3`), proj1–3 net.py:147-151 | **`{"skip": "plain"}`**(기존, `rows.json` cumulative `pbe_ebpb_plain_match`와 같은 cfg) | 없음 |
| **V2 lam0** | matching loss를 끈다(λ_s = 0) | 3b:53–81 "Training-only boundary heads", Eq. `eq:loss_s`(3b:76), `eq:objective` 𝓛 = 𝓛_seg + λ_s 𝓛_S(3b:80) | 서버 adapter.py:22-25 (`if cfg['match'] == 'none' or cfg['lambda_s'] == 0: return 0`) | **`{"lambda_s": 0}`**(기존) | 없음 |
| **V3 plainblk** | stage 1–2의 feature block eBPB를 plain block으로 바꾼다 | 3b:14–25 "Feature block", Eq. `eq:twoband`(3b:19) | net.py:55-70 `EBPB`, 교체 지점 net.py:145-146 (`make = {'ebpb': EBPB, 'mbv2': InvResBlock}[block]`; `encoder1[0][2]`, `encoder2[0][2]`) | **`{"block": "mbv2"}`**(기존) = `InvResBlock` net.py:73-85 "MobileNetV2 inverted residual at full resolution (block-ablation control for EBPB)": x + [1×1 C→C/2, BN, ReLU6, 3×3 DW, BN, ReLU6, 1×1 C/2→C, BN] | 없음 |
| **V4 segate** | channel gate(learned basis 8, coefficient dropout 0.5, 중간 비선형 없음)를 SE로 바꾼다 | 3b:25, 3b:102–104 (channel gate, "two bias-free linear layers through a learned basis of eight vectors, with dropout … no intermediate nonlinearity") | net.py:41-52 `PRCM`, 삽입 지점 net.py:113-119 `stage()` (`layers.append(PRCM(out_ch, num_basis, dropout_rate))`, :118). 9개 stage(encoder1–5, decoder1–4) 전부 | **새 key `{"gate": "se"}`**(DEFAULT_CFG에 없음 → v1 config_hash 불변) | 아래 diff |

**V3의 "plain block" 정의 (코디네이터 결정 6, 동결).** 규칙: 코드가 `mbv2`의 width 설정을 허용해 FLOPs를 feature block(eBPB)의 ±15 % 안으로 맞출 수 있으면 width를 줄이고, 아니면 `mbv2`를 그대로 쓴다. 측정(thop, MACs × 2, F2 convention `manuscript_cleanup/F2_efficiency/f2_flops.py`, 서버 e0e2f07 net.py, `abl/work/abl_flops.py` → `abl_flops_out.json`):
  - block 단위(eval, 실행 해상도): stage 1 (C 128, 256²) eBPB 0.153 G vs mbv2(t = 0.5, h = 64) 2.357 G; stage 2 (C 192, 128²) eBPB 0.083 G vs mbv2(h = 96) 1.287 G. 합 0.236 G vs 3.644 G = **15.5×**.
  - width 조절: `InvResBlock(channels, t)`에 t 인수는 있으나 config/adapter로 노출되지 않고, 코드가 hidden을 `max(8, int(C·t))`로 하한 8에 묶는다. 하한 h = 8에서도 합 0.484 G = eBPB의 2.05×다. ±15 % 안에 들려면 h ≈ 2–3(stage 1 h = 2: 0.139 G)이 필요하며, 이는 코드 하한 밖의 2–3채널 병목이다. 따라서 **코드는 FLOPs 정합 width를 허용하지 않는다 → `mbv2`를 그대로(t = 0.5) 쓴다.**
  - 전체 추론 그래프(adapter.infer, 1×3×256×256): full 3.370 GFLOPs(Table 3 값과 같음) vs mbv2 6.778 GFLOPs = **2.01×**. 추론 파라미터 1,334,337 vs 1,332,897.
  - **기록 문장(원고에도 적는다): "FLOPs about 2×, an asymmetry in favour of the replacement."** 전체 망 기준 약 2×, block 기준 15.5×다. 교체 블록이 연산을 더 쓰므로, V3에서 하락이 나오면 band 분해에 불리한 조건에서 나온 하락이다. 하락이 없으면 "파라미터가 같고 연산이 약 2×인 full-resolution block과 구분되지 않음"으로 쓴다.
  - 이 블록은 band 분해(pool → high-pass + Up(ψ))가 없는 full-resolution residual block이다. eBPB와 마찬가지로 `self.apply(init_weights)` 뒤에 붙으므로 PyTorch 기본 초기화를 쓴다(기존 코드 그대로).

**V4 최소 코드 변경 (flag 뒤, private snapshot에만; 코디네이터 결정 7, 동결).** SE hidden = **max(C/16, 8)**(정수 나눗셈), **dropout unchanged** = PRCM과 같은 coefficient dropout 0.5를 hidden 벡터에 둔다. bias 없음(PRCM과 같음), 중간 ReLU, sigmoid gate(hu2020squeeze 구조). 실제 snapshot 코드:
```python
class SEGate(nn.Module):
    def __init__(self, channels, dropout_rate=0.5):
        super().__init__()
        hidden = max(channels // 16, 8)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, channels, bias=False)
    def forward(self, x):
        h = self.coeff_dropout(self.act(self.fc1(x.mean(dim=[2, 3]))))
        return x * self.fc2(h).sigmoid()[..., None, None]
# stage(..., gate='prcm'): PRCM if gate == 'prcm', SEGate(out_ch, dropout_rate) if gate == 'se', else ValueError
# Base.__init__(self, c, num_classes=1, gate='prcm'); PBENet.__init__(..., num_classes=1, gate='prcm')
# adapter.build: PBENet(cfg['c_list'], skip=cfg['skip'], block=cfg['block'], gate=cfg.get('gate', 'prcm'))
```
  - stage별 hidden: C 128 → 8, 192 → 12, 256 → 16, 384 → 24, 512 → 32. gate 파라미터 2·C·hidden 합 **99,328**(9 gate) vs PRCM 16C 합 **38,912**. 추론 파라미터 **1,394,753** vs full 1,334,337(+60,416, +4.5 %); total 1,396,296 vs 1,335,880(smoke 실측). SE가 파라미터를 더 쓰는 비대칭도 교체 쪽에 유리하다(원고에 적는다).
- (기록) DRAFT 안(hidden 8, dropout 없음), 결정 7로 대체됨:
```python
# net.py (new class, after PRCM)
class SEGate(nn.Module):
    """Squeeze-and-excitation gate; hidden width = num_basis so the parameter count equals PRCM (16C)."""
    def __init__(self, channels, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden, bias=False); self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
    def forward(self, x):
        return x * self.fc2(self.act(self.fc1(x.mean(dim=[2, 3])))).sigmoid()[..., None, None]
# stage(in_ch, out_ch, band=False, num_basis=8, dropout_rate=0.5, gate='prcm'):
#     layers.append(PRCM(out_ch, num_basis, dropout_rate) if gate == 'prcm' else SEGate(out_ch, num_basis))
# Base.__init__(self, c, num_classes=1, gate='prcm'): pass gate=gate to all 9 stage(...) calls
# PBENet.__init__(..., gate='prcm'): self.base = Base(c, num_classes, gate=gate)
# adapter.build: return PBENet(cfg['c_list'], skip=cfg['skip'], block=cfg['block'], gate=cfg.get('gate', 'prcm'))
```
- `init_weights`(net.py:98-110)는 stage 생성 뒤 `self.apply`(net.py:132)로 적용된다. 따라서 SE의 두 Linear도 trunc_normal(0.02)로 초기화된다. PRCM의 fuser와 같은 규칙이고, basis만 randn이었다.
- gate == 'prcm'이면 새 모듈을 만들지 않으므로 RNG 소비와 초기값이 바뀌지 않는다.
- **기본 경로 동일성 검사(launch 전, 필수)**: snapshot에서 DEFAULT_CFG로 `torch.manual_seed(k)` 후 build한 state_dict가 원 코드와 key·shape·값 모두 bitwise 같아야 한다. 고정 batch forward·loss도 bitwise 같아야 한다. 기존 v1 checkpoint(cvc k1)의 strict load·fp32 val Dice는 1e-6 이내여야 한다. 하나라도 다르면 launch하지 않는다. **결과(동결 전)**: `tools/ident.py` + `ident_cmp.py`, 원본 사본 `BASE/code_orig` vs snapshot `BASE/code`: state_dict 235 tensors, 고정 batch(cvc k1 train 4장) train-mode forward·loss·grads 136개, checkpoint strict-load logits 모두 **BITWISE-IDENTICAL**, val |ΔDice| = |ΔIoU| = 0.0(값은 출력하지 않음). 로그 `BASE/smoke/ident_cmp.log`.
- **rows.json** (snapshot의 `configs/rows.json`에 suite 추가, 기존 suite는 바꾸지 않음):
  `"abl4": {"pbe_idskip": {"model":"pbe","cfg":{"skip":"plain"}}, "pbe_lam0": {"model":"pbe","cfg":{"lambda_s":0}}, "pbe_mbv2": {"model":"pbe","cfg":{"block":"mbv2"}}, "pbe_se": {"model":"pbe","cfg":{"gate":"se"}}}`
  main.py:87 `cfg = copy.deepcopy(ad.DEFAULT_CFG); cfg.update(r['cfg'])`이 그대로 병합한다.

**one-factor 확인(사전 기록)**
- V1: proj1–3은 남지만 forward에서 쓰이지 않는다(grad None이어서 AdamW가 갱신하지 않음). shead1–3과 matching loss는 s_i = e_i 위에서 그대로 작동한다.
- V2: shead는 남지만 loss에 들어가지 않는다. deep supervision(gt_conv1–4)은 그대로다.
- V3: block만 바뀐다. stride·gate·skip은 같다.
- V4: gate만 바뀐다.
- 네 변형 모두 추론 그래프의 나머지는 같다. 추론 파라미터 수(`params_inference`)를 run별로 보고한다.

## 3. 실행
- `BASE=<WORK>/abl`(archive 밖). snapshot = `BASE/code`(e0e2f07 복사 + §2 변경, `__pycache__` 제외). 변경 파일과 snapshot digest를 `BASE/SHA256SUMS`에 쓰고, queue 시작 시 확인한다.
- 명령(run마다, cwd `BASE/code`, `pbe_v1_fill/PLAN.md` 형식):
  `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 0-7,12-31 python main.py --suite abl4 --rows <row> --datasets <D> --splits <K> --stage train --execute --device cuda:0 --data-root <DATA> --cache-root <ARCHIVE>/cache --run-root BASE/rr`
- 60 runs = 4 변형 × 3 ds × 5 split. **순서**: ds 비용이 낮은 순(cvc → isic17 → isic18), 그 안에서 변형 순, 그 안에서 k = 1..5다. cap으로 중단되더라도 (변형, ds) 5-split 묶음이 최대한 완성된다.
- 전처리 검사(run마다): 원 캐시 3개가 존재하고 이름이 full v1 run의 `data_cache`와 같아야 한다. 아니면 FAIL로 표시한다(캐시를 새로 쓰지 않음).
- 사후 검사(run마다, `tools/verify_abl.py post`): status complete, `config.model` == pbe, `model_cfg` == DEFAULT_CFG ∪ 변형 cfg, protocol == full v1 run, seed = split = k, config_hash 재현, history 100행, `data_cache` == full v1 run, `code.source_sha256` == snapshot digest, `test == "SEALED"`, checkpoint 존재. 실패하면 MISMATCH로 표시하고 결과는 보존한다. test IoU 유한성은 eval_test 단계에서 확인한다.
- snapshot: `BASE/code` = e0e2f07 사본 + 4 파일 변경(`models/pbe/src/net.py`, `models/pbe/adapter.py`, `configs/rows.json`, `common/trainer.py`), `DEPLOYED_COMMIT` = `e0e2f07c…+F4abl`, source digest **`ee357c57035bad31fffbfb18832e2f9f235fac1234616804fef1719719dd022f`**. 원본 사본 `BASE/code_orig` digest 496cc2db…bf7(= v1 runs).
- `flock BASE/gpu1.lock`, 단일 queue(`tools/run_abl.sh 1`, nohup setsid), 재시작 안전(DONE은 건너뜀). 실패 시 1회 자동 재시도하고, 이전 시도는 `<run>_attempt<n>`으로 옮긴다(삭제하지 않음). 2회 실패는 `FAIL_FINAL`. marker: run 폴더 `BASE/rr/abl4/<row>/<ds>/k<k>/{DONE,FAIL,FAIL_FINAL,MISMATCH}`, queue 끝 `BASE/ALL_DONE`.
- GPU 1만 쓴다(F3은 GPU 0). 종료는 PID로만 한다.
- **stop rules**: (a) cap guard: 누적 GPU-h(`BASE/gpuh_ledger.tsv`, 시도별 wall 시간, 실패 포함, 동결 전 smoke·timing 0.15 h 포함) + 다음 run 추정(평균 v1 train_minutes(ds) × 측정 배율(v) × 1.25 + 3 min)이 60 − 1.0(run_decide 예비) GPU-h를 넘으면 그 run을 시작하지 않고 `BASE/STOPPED_CAP`을 쓴 뒤 ALL_DONE(stopped=cap)으로 끝낸다. (b) OOM(로그의 "out of memory") 2회 → `BASE/STOPPED_OOM`, ALL_DONE을 쓰지 않고 멈춘다(재시작도 거부). (c) GPU 1에 다른 compute process가 있으면 run 시작 전 60 s 간격으로 기다린다.
- **official test 봉인**: 학습 중 test를 평가하지 않는다. snapshot `common/trainer.py`에 env flag `ABL_SEAL_TEST=1`일 때만 test 평가를 건너뛰고 `results.json` `test = "SEALED"`로 쓰는 분기를 넣었다(flag가 없으면 원 경로 그대로). checkpoint 규칙(max val mean IoU)은 그대로다. test는 `tools/run_decide.sh`가 **ALL_DONE 뒤 1회** `tools/eval_test.py`로 평가한다(trainer와 같은 경로: best checkpoint strict load, `common.metrics.evaluate`, batch 1, threshold 0.5; `test_eval.json`이 있으면 거부). run_decide.sh는 ALL_DONE이 없거나, queue lock이 잡혀 있거나, `abl_result.json`이 있으면 거부한다.

## 4. 판정 (결과 전 고정)
- **지표**: run별 test mean IoU. 변형은 `test_eval.json` `test.iou`(봉인 해제 1회), full은 `results.json` `test.iou`. Table 1과 같은 필드이고 같은 계산 경로다(`tools/sig_table1.py` = `code/pbe_seg/tools/make_tables.py` 재현). Dice는 보고만 한다.
- **검정**: (변형 v, ds d)마다 v의 split k와 full v1의 split k를 짝짓는다(k = 1..5). 양측 paired t-test(`scipy.stats.ttest_rel`, df = 4)다.
- **다중비교**: 4 변형 × 3 ds = **12개 p값 전체에 Holm step-down**, α = 0.05.
- **유의한 하락** ⇔ Holm 보정 p < 0.05 이고 평균 차(v − full) < 0.
- **"구성요소에 효과가 있다"** ⇔ 3개 ds 중 **2개 이상에서 유의한 하락**.
- **사전에 정한 결과 처리**:
  - 효과가 있는 구성요소가 **0개**이면 제목과 기여 목록에서 구성요소 주장을 지운다. "Predictive-Residual Skip Connections" 같은 구성요소 이름을 효과 주장으로 쓰지 않는다. Method는 구성요소를 효과 주장 없이 기술하고, Results는 ablation을 null로 보고한다.
  - 효과가 있는 구성요소가 **1개 이상**이면 그 구성요소만 "protocol v1에서 측정된 효과"로 쓸 수 있다. 나머지는 효과 주장 없이 기술한다.
  - 어떤 변형이 2개 이상 ds에서 **유의하게 높으면**(평균 차 > 0) 그 구성요소를 Results와 Limitations에 "해롭다"로 보고한다.
  - 하락이 유의한 ds가 1개뿐이면 "효과 주장 없음"으로 두고 수치를 보고한다.
- **보고 표**(12셀): 평균 차, 쌍별 차의 t 기반 95 % CI, raw p, Holm p, 유의 표시, 변형의 mean ± SD, full의 mean ± SD, Dice 차, `params_inference`.
- cap이나 FAIL로 5 split이 다 차지 않은 (v, d)는 "검정 안 됨"으로 둔다. 효과 없음으로 세지 않는다. Holm family는 검정된 p값 수로 줄이고, 그 수를 보고한다.

## 5. 합성 역검사 (분석 스크립트 `abl_analyze.py`, 실데이터 전)
1. **실데이터 경로 재현**: 같은 함수로 Table 1 기존 값 1개를 재현한다. CVC PBE-Net vs MK-UNet은 raw p 0.0075, 평균 차 +0.023이어야 하고(`TABLE1_SIG.md`), 기존 결과 파일만 읽는다. 이 검사는 필드·짝짓기·부호 규약을 확인한다.
2. **planted drop**: 변형 V1 = full − 0.05 + N(0, 0.005)를 cvc·isic17에 넣고 나머지는 full + N(0, 0.005)로 둔다. 판정은 V1 "효과 있음", 나머지 "효과 없음"이어야 한다.
3. **null**: 모든 변형을 full + N(0, SD_full)로 둔 family 1000개(seed 7919)에서, 효과 있음 구성요소가 1개 이상 나오는 비율은 ≤ 0.05여야 한다. Holm의 FWER 보장을 확인한다.
4. **부호**: 변형 = full + 0.05이면 "유의하게 높음"으로 분류되어야 하고 "하락"으로 분류되면 안 된다.
5. **Holm·t 수식**: 손으로 계산한 12개 p 사례와 일치해야 한다. `ttest_rel`과 수동 t 공식의 p 차가 < 1e-12여야 한다(서버 env에서 scipy를 import할 수 없으면 수동 공식을 쓰고 이 사실을 기록한다).
6. **불완전 묶음**: 4 split만 있는 (v, d)는 "검정 안 됨"으로 처리되고 family 크기가 11로 줄어야 한다.
출력은 "SYNTHETIC" 접두사로 `BASE/synthetic.log`에 쓴다. 하나라도 FAIL이면 launch하지 않는다.
**결과(동결 전, 서버 env scipy 1.10.1)**: 1 PASS(p = 0.0075, diff +0.0227), 2 PASS(V1만 effect, family 12), 3 PASS(1000 null family: P(≥1 effect) = 0.000, P(Holm 유의 셀 ≥1) = 0.042), 4 PASS(전부 harmful, drop 0), 5 PASS(Holm 손계산 일치, |p_manual − p_scipy| ≤ 4.4e-16), 6 PASS(family 11). SYNTHETIC ALL PASS.

## 6. sha 기록·결과 1회
- 동결 시 다음 sha256을 `BASE/SHA256SUMS`에 기록한다(queue·run_decide 시작 시 `sha256sum -c`): snapshot 변경 파일 4개(`code/models/pbe/src/net.py`, `code/models/pbe/adapter.py`, `code/configs/rows.json`, `code/common/trainer.py`)와 그 원본(`code_orig/…`, e0e2f07), `tools/` 전부(`abl_common.py`, `abl_analyze.py`, `verify_abl.py`, `eval_test.py`, `run_abl.sh`, `run_decide.sh`, `smoke_variants.py`, `ident.py`, `ident_cmp.py`, `abl_flops.py`), 이 prereg, full v1 15 run(선택 3 ds)의 `results.json`.
- smoke와 timing은 loss·grad norm 유한 여부와 ms/step만 출력하고, val/test 지표는 출력하지 않는다.
- 60 runs가 DONE·FAIL·MISMATCH로 모두 확정된 뒤 `abl_analyze.py`를 **1회** 실행한다. 학습 run의 재실행은 crash·FAIL 재시도만 허용한다. 결과를 본 뒤 run을 추가하거나 교체하지 않는다. 버그 수정은 AMENDMENT로만 한다.

## 7. 런타임 추정 (GPU 1, cap 60 GPU-h)
- 기존 PBE v1 `train_minutes`(서버 results.json 30개, 2026-10-16 확인): cvc 평균 13.6 [8.9–18.1], isic17 43.4 [35.0–51.9], isic18 57.1 [52.6–59.8], (참고 PH2 3.8, etis 4.2, kvasir 21.7).
- split 하나에 3 ds를 도는 시간: 평균 114.0 min, 최댓값 기준 129.8 min.
- 변형별 시간 배율(timing gate 전 가정): V1, V2, V4는 1.0(V2는 shead loss가 빠져 조금 짧을 수 있음). V3 mbv2는 stage 1–2 full-resolution 연산이 늘어 1.5(중심)–2.0(상한)으로 둔다.
- **추정**: 중심 5 split × 114.0 min × (1 + 1 + 1.5 + 1) / 60 = **42.8 GPU-h**에 run마다 start-up·test 약 1 min(1.0 h)과 smoke·timing 0.3 h를 더해 **≈ 44 GPU-h**다. 상한 5 × 129.8 × 5.0 / 60 + 1.3 = **≈ 55.4 GPU-h**다. wall은 GPU 1 단독으로 약 2일이다.
- **Timing gate 결과(동결 전, GPU 1 RTX 3090, `tools/smoke_variants.py`, isic18 k1 train batch 16, warm-up 5 + 50 step, batch는 GPU에 미리 올림)**: full 153.6 ms/step; 배율 idskip 0.962, lam0 0.994, mbv2 1.058, se 1.015(peak mem mbv2 9.8 GB, 나머지 7.1 GB). 4 변형 × 55 step 모두 loss·grad norm 유한, 교체 모듈 확인(idskip: proj grad None; lam0: extra loss 0, shead grad None; mbv2: stage 1–2 InvResBlock; se: SEGate 9, PRCM 0). **투영** = 5 × 114.1 min × Σ_v ratio 4.029 / 60 = 38.3 GPU-h + start-up 1.0 + smoke·timing 0.15 + run_decide 약 0.5 → **≈ 40 GPU-h**. 상한(최대 train_minutes 합 129.8 min, mbv2 배율 1.5 가정) = 5 × 129.8 × 4.471 / 60 + 1.65 = **≈ 50 GPU-h** ≤ 55 → launch.
- **Timing gate**(`C_PLAN.md` §4 방식): isic18 k1 train cache에서 변형별로 warm-up 5회 후 50 step의 ms/step을 full과 비교한다. 투영 = Σ_(v,d) 5 × mean train_minutes(d) × ratio(v) + 1.3 h다. **투영이 55 GPU-h 이하일 때만 launch한다.** 넘으면 launch하지 않고 코디네이터에게 돌려준다(A1의 대안 block으로 바꾸는 것 등). 실행 중 누적 60 GPU-h에 도달하면 queue를 멈추고 §4의 "검정 안 됨" 규칙을 따른다.

## 8. 코디네이터 결정 (동결 시 해결; 아래 원 선택지는 기록으로 남김)
- A1 → 결정 6: width 정합 불가(하한 h = 8에서도 2.05×) → `mbv2` 그대로, "FLOPs about 2×, an asymmetry in favour of the replacement" (§2).
- A2 → train 수와 전체 수 모두 isic18 최대(§1). A3 → isic17 유지, "two ISIC datasets included by the fixed rule" (결정 8).
- A4 → 결정 7: SE hidden max(C/16, 8), dropout 0.5 유지, bias 없음.
- A5 → `lambda_s: 0`. A6 → 양측(결정 규칙 유지). A7 → F3(GPU 0)와 병행 launch, §4 결과 처리 문구는 그대로.

### (기록) 동결 전 미해결 선택
- **A1 plain block 정의**: 제안은 기존 key `block: mbv2`다(코드에 있는 block-ablation 대조군, 파라미터 거의 같음, FLOPs 약 2배). 대안 (a) ψ(x)를 full resolution에 적용(= EBPB `branch3`를 pool 없이 적용, 파라미터 동일, 새 key 필요). (b) block 없음(nn.Identity, compute-matched, 파라미터 감소, 새 key 필요).
- **A2 "가장 큰 데이터셋" 기준**: 제안은 train 영상 수(캐시 크기 추정 isic18 약 1618 > isic17 약 1290)다. 동결 전 정확한 수를 확인한다. 전체 영상 수 기준이어도 isic18이 클 가능성이 높지만 확인이 필요하다.
- **A3 무작위 결과 isic17**: 두 ISIC가 함께 들어가 피부 영역 2개, 내시경 1개가 된다. seed 규칙은 고정이므로 바꾸지 않는 것이 제안이다. 바꾸려면 동결 전에 규칙 자체를 바꿔야 한다(예: "isic17 제외"를 미리 명시).
- **A4 SE 세부**: hidden 8(파라미터 동일, 제안) vs C/16(표준 축소비). dropout 없음(표준 SE, 제안) vs PRCM과 같은 coefficient dropout 0.5 유지(비선형·초기화만 분리).
- **A5 λ_s = 0 vs `match: none`**: 서버 adapter.py:23에서 둘 다 loss 0이다. config 표기로 `lambda_s: 0`(코디네이터 문구)을 제안한다.
- **A6 검정 방향**: Table 1과 같은 양측 검정(제안) vs 하락 단측 검정.
- **A7 F3 결과 의존**: SUMMARY §6은 F4를 F3 뒤에 두었다. F3 분기 (ii)가 "artifact"이면 제목 재결정과 함께 F4의 결과 처리 문구(§4)를 다시 확인할지.

## 9. 위험·한계 (사전 기록)
- n = 5 split이라 검정력이 낮다. 효과 없음은 동등함을 뜻하지 않는다. TABLE1_SIG 기준 PBE-Net split SD는 cvc 0.027, isic17 0.006, isic18 0.006이다. isic 계열에서는 작은 차도 유의할 수 있고, cvc에서는 큰 차가 필요하다.
- seed가 split과 묶여 있다(seed = k). 쌍별 차에는 split과 seed 효과가 함께 들어 있다(R3 지적, protocol v1의 성질).
- GPU backward 비결정성으로 같은 cfg도 bitwise 재현되지 않는다. full v1 run은 RTX 3090(GPU 0 또는 1)에서 학습되었다.
- V3 mbv2는 FLOPs가 다르다. 정확도 차를 "band 분해의 효과"로 읽을 때 연산량 차이를 함께 적는다.

## AMENDMENT
(없음. 동결 이후 변경은 여기에 시각·사유와 함께 적는다.)
