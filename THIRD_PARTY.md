# Third-party baseline sources (not redistributed)

The adapters in experiments/qfix/A/code/models/<model>/ load these files from src/. They are byte-identical copies of the public implementations of each baseline that were used for every number of the manuscript. Obtain each implementation from its authors, place the file at the path below, and check its sha256 (also recorded in models/<model>/manifest.json).

| Path | Bytes | sha256 |
|---|---|---|
| experiments/qfix/A/code/models/cmunext/src/CMUNeXt.py | 6029 | cae1d28e7de924fe6f8d667211a498489a6f9765d6a0cbb986b43aeca9ba3f55 |
| experiments/qfix/A/code/models/egeunet/src/EGEUNet.py | 14055 | 3dd2b6817618843d95910cc4348640a8636155770a1f04a9d31bf83392924ace |
| experiments/qfix/A/code/models/lbunet/src/LBUNet.py | 12896 | 61ff2ba8d5172f27eec8603565a087cdfbc1605b12ebee8b2ccac73df66d3ed6 |
| experiments/qfix/A/code/models/malunet/src/MALUNet.py | 12680 | 7bab73227cd159d6a82ee0df6ad406352ce54f0a578c38487753605e34caa739 |
| experiments/qfix/A/code/models/maresunet/src/MAResUNet.py | 7690 | 8df94950a4fa9008b0f1ca251953ee3e37ec119cc18d5e6a3dc886b82d71b30d |
| experiments/qfix/A/code/models/mhorunet/src/MHorUNet.py | 17899 | df5b9459b07bf7af913072975ac7591c5ca0b10cdcc02649fb033ebf8d4afb09 |
| experiments/qfix/A/code/models/mk_unet/src/MK_UNet.py | 25825 | 11c25ca612835b058b012689572340d020773440b87d478ebb4e2c2922a92ac1 |
| experiments/qfix/A/code/models/tinyunet/src/TinyUNet.py | 4761 | 109c4b371d4b08add6fec1df45032d6ff6ad1d71b6e8ba8e586cf4d72bd55422 |
| experiments/qfix/A/code/models/ucmnet/src/UCMNet.py | 19852 | 60b0493d70fb9dbda4badf1a92ab89e787c9c321d50fc5c63226f5c745b95904 |
| experiments/qfix/A/code/models/utanet/src/UTANet.py | 19069 | 33b47f16f4787c13299cfd0e06fd48f5fa16880e85baf9296495353bd2e56066 |

PyPI packages: ptflops 0.7.5 (FLOP counts of Table 2), thop (FLOP counts of the ablation).
