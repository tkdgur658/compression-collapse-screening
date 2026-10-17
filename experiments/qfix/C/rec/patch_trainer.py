p = "common/trainer.py"
s = open(p).read()
reps = [
("def run(adapter, model_cfg, dataset, k, out_dir, data_root, cache_root, device='cuda:0', protocol=None):\n",
 "def run(adapter, model_cfg, dataset, k, out_dir, data_root, cache_root, device='cuda:0', protocol=None,\n"
 "        skip_test=False):\n"
 "    # skip_test (QFIX stage C, default False = protocol v1 unchanged): the official test split stays sealed in the\n"
 "    # training job; no test loader is built and no test metric is computed (evaluated once later by the stage-C evaluator).\n"),
("    te = DataLoader(D.SegCache(caches['test'], False), batch_size=1, shuffle=False, num_workers=2)\n",
 "    te = None if skip_test else DataLoader(D.SegCache(caches['test'], False), batch_size=1, shuffle=False, num_workers=2)\n"),
("    test, per_image = evaluate(model, te, adapter, device)\n"
 "    print(f'[test] iou={test[\"iou\"]:.4f} dice={test[\"dice\"]:.4f} (best ep {best_ep}, {train_min:.1f} min)', flush=True)\n",
 "    if skip_test:\n"
 "        test, per_image = None, None\n"
 "        print(f'[sealed] test not evaluated (best ep {best_ep}, {train_min:.1f} min)', flush=True)\n"
 "    else:\n"
 "        test, per_image = evaluate(model, te, adapter, device)\n"
 "        print(f'[test] iou={test[\"iou\"]:.4f} dice={test[\"dice\"]:.4f} (best ep {best_ep}, {train_min:.1f} min)', flush=True)\n"),
("           'test_per_image_iou': per_image, 'test_names': D.SegCache(caches['test'], False).names,\n",
 "           'test_per_image_iou': per_image, 'test_names': [] if skip_test else D.SegCache(caches['test'], False).names,\n"),
("    out['test_names'] = [str(n) for n in out['test_names']]\n",
 "    out['test_names'] = [str(n) for n in out['test_names']]\n"
 "    if skip_test:\n"
 "        out['test_sealed'] = True\n"),
]
for a, b in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched")
