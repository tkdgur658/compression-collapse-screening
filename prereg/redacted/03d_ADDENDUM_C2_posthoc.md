# PREREG_QFIX — 사후 부록 C2 (사후 튜닝(사전등록 밖))

**표기: 사후 튜닝(사전등록 밖).** 이 부록은 PREREG_QFIX.md(수정 없음)의 사전등록 분기 밖이다. 단계 C 판정 (i)–(iii)에는 넣지 않는다. 원고에서는 "post hoc" 단락으로만 분리 보고한다. 선행 = DFQ/CLE(Nagel et al. 2019), 방법 기여 = 교과서.

- 작성 2026-10-16 04:3x KST. C1 판정(TRADE-OFF: (i) 15/18, (ii) 0/6, (iii) AUC .978) 뒤 study lead 요청으로 실행했다.
- 명세 `B_fix_design.md` §8, 계획 `C2/C2_PLAN.md`(측정 전 작성), 스크립트 `C2/qfix_c2_cle.py` sha256 fed02091…2ada. Q 프로토콜은 동결 `eval_C1.py`(sha256 152694ec…1166)를 import해서 그대로 썼다.
- 변형 1종: stem 1×1 conv(3→128) → RepConv(depthwise k×k·1×1, bn_identity)에만 CLE를 적용했다. s_c = sqrt(r2_c/r1_c)이고 학습은 없다. record 양자화기는 바꾸지 않았다. bn_identity는 running_mean ×s, γ ÷s로 처리해 정확히 보존했다(§8의 var ×s² 근사 대신 쓴 동치 구현).
- 함수 보존: 상대 logit 차 최대 1.4e-6(게이트 1e-4, 60/60 통과), fp32 val Dice 변화 최대 4.2e-7, C1 test Dice 변화 최대 1.0e-6.
- (a) v1+CLE: 0/18 셀(v1 0/18). setting별 평균 R은 w4 .477(v1 .421), a6 .303(.300), l50 .102(.049)이다. AUC .973.
- (b) C1+CLE: 12/18 셀(C1 15/18). 평균 R은 w4 .840(C1 .934), a6 .856(.875), l50 .774(.893)로 모두 C1보다 낮다. AUC .980. fp32 test Dice는 C1과 같아서 (ii) 결과(0/6)는 그대로다.
- 해석(사후): stem CLE는 Q 붕괴를 풀지 못했다. v1에서는 개선이 미미하고, C1에서는 오히려 악화됐다. C1 결론(정확도-강건성 트레이드오프, 주장 없음)은 바뀌지 않는다. GPU 0.14 h(gpu-server GPU 0).
