"""Write the sweep 12 one-cell Colab launchers with the current pipeline/ code embedded: colab/connectome_sweep12.ipynb
(Drive mounted in the Colab account) and colab/connectome_sweep12_rclone.ipynb (any Colab account, your Drive via
rclone). Sweep 12 = better group fine-tunes on validation (WORKSPACE.md "Sweep 11 results" and the LB probes after it).

    python colab/build_sweep12.py
"""
import os
import pprint
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sweep11 as b11  # the shared pieces: Drive / rclone setup, data download, pipeline files, train()

HEADER = '''# Alpha Connectome, sweep 12: better group fine-tunes on validation. One A100 run (~1-1.5 h incl. the 30-min data
# download), self-contained (the whole pipeline is embedded below).
# Starting point: sweep11_mix.zip, LB 0.66021 = groups 1-3 fine-tuned on all of validation (sweep11_final/vg*_lr2_u06_all,
#   hedge 0.9), group 4 = sweep 10a's ft_film~wise. Fine-tuning group 4 on validation lost on the LB (0.65956);
#   full training on validation memorises the sequences (WORKSPACE.md). Inputs from Drive: sweep9/s9_h128,
#   sweep10a/ft_g1..3, sweep10a_routed/submission, sweep11_final/vg*_all, cache/t_9.npy, cache/mask_train.npy,
#   cache/tv_s9.npy, mask_model/split.json.
# 12a fine-tunes (-> runs/sweep12_ft, each on half A and on half B), groups 1-3 from the 10a specialists:
#   vgN_ref (= sweep 11's recipe: lr 2e-4, 3 passes, unscored rows 0.06, average of passes 2-3), vgN_u0 / vgN_u02
#   (unscored rows 0 / 0.02), vgN_e6 (6 passes), vgN_lr1 (lr 1e-4), vgN_head (output layer only, lr 1e-3),
#   vgN_soup (4 fine-tunes on random 75% shares, weights averaged = one model), vgN_mix (validation + 2000 train
#   sequences with teacher targets in the same batches); group 4 from s9_h128: vg4_head, vg4_top (output layer + last
#   GRU layer), vg4_light (lr 5e-5, one pass, half the loss on s9_h128's own predictions), vg4_mix.
# 12b cross-fit table (runs/sweep12_crossfit.json): each model scored on the half it did not train on, at 0.9 / 0.1
#   weights with its start (as submitted). Per group the best recipe replaces sweep11_mix's model only if it beats
#   vgN_ref (groups 1-3, by > 0.0002) or s9_h128 (group 4, by > 0.0005).
# 12c the winners retrained on all of validation (runs/sweep12_final) -> zips/sweep12_routed.zip (+ sweep12_g13.zip /
#   sweep12_g4.zip when group 4 changed, to tell the parts apart on the LB); speed test.
# Re-runnable: finished models are skipped. Keep this tab open while it runs.
'''

PULL = ("PULL = ['sweep9/summary.jsonl', 'sweep9/s9_h128/**', 'cache/t_9.npy', 'cache/mask_train.npy', 'mask_model/split.json',\n"
        "        *[f'sweep10a/ft_g{g}/best.pt' for g in (1, 2, 3)], 'sweep10a_routed/submission/**',\n"
        "        *[f'sweep11_final/vg{g}_lr2_u06_all/best.pt' for g in (1, 2, 3)],\n"
        "        'cache/tv_s9.npy', 'sweep12_*/**', 'sweep12_crossfit.json']\n"
        "PUSH = ['sweep12_*/**', 'sweep12_crossfit.json', 'zips/sweep12*']\n")

SWEEP12 = r'''
# ======== sweep 12
SPEC = [f'{OUT_DIR}/sweep10a/ft_g{g + 1}/best.pt' for g in range(3)]  # sweep 10a group specialists: the fine-tunes' start
S11 = [f'{OUT_DIR}/sweep11_final/vg{g + 1}_lr2_u06_all/best.pt' for g in range(3)]  # groups 1-3 of sweep11_mix.zip
G4_ONNX = f'{OUT_DIR}/sweep10a_routed/submission/model_4.onnx'  # ft_film~wise: group 4 of sweep11_mix.zip
S9_ONNX = f'{OUT_DIR}/sweep10a_routed/submission/model_0.onnx'  # s9_h128: the fallback (unsure group detection)
for need in SPEC + S11 + [G4_ONNX, S9_ONNX]:
    assert os.path.exists(need), f'missing in Drive: {need} (sweeps 10a / 11)'
TV = '/content/wnn/tv_s9.npy'  # s9_h128's predictions on valid (the anchor of vg4_light)
if not os.path.exists(TV):
    if os.path.exists(f'{CACHE}/tv_s9.npy'):
        sh(f'cp {CACHE}/tv_s9.npy {TV}')
    else:
        sh(f'cd /content/wnn && python -m pipeline.teacher --ckpts {S9} --data datasets/valid.parquet --out {TV} && cp {TV} {CACHE}/tv_s9.npy')
FLAGS_V = f'--pool valid --teacher-valid {TV} --evals-per-epoch 0 --skip-final-eval'
FLAGS_M = f'--pool train+valid --train-seqs 2000 --teacher-valid {TV} --evals-per-epoch 0 --skip-final-eval'

def ftv(name, part, g, **kw):  # sweep 11's group fine-tune (in sweep11_mix.zip): group g, validation part A | B | all
    return {'name': f'{name}_{part}', 'init': SPEC[g] if g < 3 else S9, 'group': g, 'lr': 2e-4, 'epochs': 3,
            'sched': 'const', 'select': False, 'swa_from': 2, 'valid_part': part, 'unscored_weight': 0.06, **kw}

RECIPES = {}  # name -> (kind, group, settings); kind: valid (validation pool) | soup (bagged members) | mixed (+ train)
for g in range(3):
    k = f'vg{g + 1}'
    RECIPES.update({f'{k}_ref': ('valid', g, {}), f'{k}_u0': ('valid', g, {'unscored_weight': 0.0}),
                    f'{k}_u02': ('valid', g, {'unscored_weight': 0.02}), f'{k}_e6': ('valid', g, {'epochs': 6, 'swa_from': 4}),
                    f'{k}_lr1': ('valid', g, {'lr': 1e-4}), f'{k}_head': ('valid', g, {'train_part': 'head', 'lr': 1e-3}),
                    f'{k}_soup': ('soup', g, {'valid_frac': 0.75}),
                    f'{k}_mix': ('mixed', g, {'distill': 1.0, 'teacher': T9, 'mask_weight': 0.5})})
RECIPES.update({'vg4_head': ('valid', 3, {'train_part': 'head', 'lr': 1e-3}), 'vg4_top': ('valid', 3, {'train_part': 'top'}),
                'vg4_light': ('valid', 3, {'lr': 5e-5, 'epochs': 1, 'swa_from': 1, 'distill': 0.5, 'teacher': T9, 'valid_distill': 1.0}),
                'vg4_mix': ('mixed', 3, {'distill': 1.0, 'teacher': T9, 'mask_weight': 0.5})})
N_BAG = 4

def members(name, part):
    kind, g, kw = RECIPES[name]
    if kind == 'soup':
        return [ftv(f'bag_{name}_k{k}', part, g, bag_seed=k, **kw) for k in range(N_BAG)]
    return [ftv(name, part, g, **kw)]

def run_recipes(names, parts, out):
    """Train the recipes on the validation parts, then make the soups."""
    v = [c for p in parts for nm in names if RECIPES[nm][0] != 'mixed' for c in members(nm, p)]
    m = [c for p in parts for nm in names if RECIPES[nm][0] == 'mixed' for c in members(nm, p)]
    if v:
        train(v, out, epochs=max(c['epochs'] for c in v), flags=FLAGS_V)
    if m:
        train(m, out, epochs=max(c['epochs'] for c in m), flags=FLAGS_M)
    for p in parts:
        for nm in names:
            dst = f'{out}/{nm}_{p}/best.pt'
            if RECIPES[nm][0] == 'soup' and not os.path.exists(dst):
                sh(f"cd /content/wnn && python -m pipeline.soup --out {dst} " + ' '.join(f"{out}/{c['name']}/best.pt" for c in members(nm, p)))
    sync_up()

# ---- 12a: every recipe on half A and on half B
SW = f'{OUT_DIR}/sweep12_ft'
run_recipes(list(RECIPES), 'AB', SW)

# ---- 12b: cross-fitted scores (each model on the validation half it did not train on)
CF = f'{OUT_DIR}/sweep12_crossfit.json'
if not os.path.exists(CF):
    paths = ' '.join(f'"{SW}/{nm}_A/best.pt"' for nm in RECIPES)
    sh(f'set -o pipefail; cd /content/wnn && python -m pipeline.crossfit score --valid-split {SPLIT} --base {S9} --base-spec {" ".join(SPEC)} '
       f'--recipes {paths} --hedge 0.9 --out {CF} 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
    sync_up()
else:
    print(f'cross-fit table already made: {CF}', flush=True)
sc = json.load(open(CF))['scores']
H = '~0.9'  # as submitted: 0.9 fine-tune + 0.1 its start
gs = lambda nm, g: sc.get(nm + H, {}).get(f'g{g + 1}', float('nan'))
chosen = [None] * 4  # recipe name per group, None = keep sweep11_mix's model
print('\ncross-fitted scored-row WP per group (hedged 0.9; d = vs the model in sweep11_mix.zip: vgN_ref for groups 1-3, '
      's9_h128 for group 4):', flush=True)
for g in range(4):
    ref_name = f'vg{g + 1}_ref' if g < 3 else None
    ref = gs(ref_name, g) if ref_name else sc['s9 (base)']['g4']
    names = sorted((nm for nm, r in RECIPES.items() if r[1] == g), key=lambda nm: -(gs(nm, g) if gs(nm, g) == gs(nm, g) else -1))
    base = sc['s9 (base)'] if g == 3 else sc.get('base routed (spec g1-3, s9 g4)', {})
    print(f"  group {g + 1}: before any validation fine-tune ({'s9_h128' if g == 3 else '10a specialist'}) "
          f"{base.get(f'g{g + 1}', float('nan')):.4f}", flush=True)
    for nm in names:
        print(f'    {nm:12s} {gs(nm, g):.4f}  d {gs(nm, g) - ref:+.4f}', flush=True)
    best = names[0] if names else None
    margin = 0.0002 if g < 3 else 0.0005
    if best and best != ref_name and gs(best, g) > ref + margin:
        chosen[g] = best
print('chosen per group (None = keep sweep11_mix.zip):', chosen, flush=True)

# ---- 12c: the winners on all of validation -> submissions
SW_FIN = f'{OUT_DIR}/sweep12_final'
winners = [nm for nm in chosen if nm]
os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
SUBS = {}
if not winners:
    print('\ndecision: no recipe beat sweep11_mix.zip by the margin -> nothing new to submit; sweep11_mix.zip stays.', flush=True)
else:
    run_recipes(winners, ['all'], SW_FIN)
    new = [f'{SW_FIN}/{chosen[g]}_all/best.pt' if chosen[g] else (S11[g] if g < 3 else G4_ONNX) for g in range(4)]
    old = S11 + [G4_ONNX]
    variants = {'routed': new}
    if chosen[3] and any(chosen[:3]):  # tell the group 4 change and the groups 1-3 changes apart on the LB
        variants['g13'] = new[:3] + [G4_ONNX]
        variants['g4'] = old[:3] + [new[3]]
    for tag, route in variants.items():
        sub = f'{OUT_DIR}/sweep12_{tag}/submission'
        if not os.path.exists(f'{sub}/routes.json'):
            sh(f'set -o pipefail; cd /content/wnn && python -m pipeline.crossfit export --route {" ".join(route)} --fallback {S9_ONNX} '
               f'--alpha 0.9 --out {sub} 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
        SUBS[tag] = sub
        print(shutil.make_archive(f'{OUT_DIR}/zips/sweep12_{tag}', 'zip', sub), flush=True)
    sync_up()
    print('\ndecision: per group, the cross-fitted change vs sweep11_mix.zip:', flush=True)
    for g in range(4):
        if chosen[g]:
            ref = gs(f'vg{g + 1}_ref', g) if g < 3 else sc['s9 (base)']['g4']
            print(f'  group {g + 1}: {chosen[g]} {gs(chosen[g], g):+.4f} vs {ref:.4f} ({gs(chosen[g], g) - ref:+.4f})', flush=True)
    print('  Submit zips/sweep12_routed.zip first (LB to beat: 0.66021). With a group 4 change, sweep12_g13.zip /\n'
          '  sweep12_g4.zip show which part moved the LB (sweep 11: g1-3 gains transferred ~3x, the g4 loss did too).', flush=True)
    sh(f"cd /content/wnn && python -m pipeline.speed {' '.join(SUBS.values())} {OUT_DIR}/sweep9/s9_h128/submission")  # grader ~ 0.52 x this
sync_up()
print(f"done; results in Drive: {globals().get('REMOTE', OUT_DIR)}", flush=True)
'''


def main():
    rclone = b11.RCLONE
    start, end = rclone.index("PULL = ["), rclone.index("inc = lambda")
    rclone = rclone[:start] + PULL + rclone[end:]
    body = b11.BODY[:b11.BODY.index("# ---- Phase A")]
    for a, b in [("2026-11-05", "2026-11-08"),
                 ("assert 'valid_part' in ExpConfig.__dataclass_fields__", "assert 'train_part' in ExpConfig.__dataclass_fields__"),
                 ("import pandas as pd\n", "")]:
        assert a in body, a
        body = body.replace(a, b)
    body = body[:body.index("def table(")] + SWEEP12
    files = {f"pipeline/{os.path.basename(p)}": open(p).read()
             for p in sorted(__import__("glob").glob(os.path.join(b11.ROOT, "pipeline", "*.py")))}
    b11.FILES.clear()
    b11.FILES.update(files)
    body = body.replace("__FILES__", pprint.pformat(files, width=120))
    b11.write("connectome_sweep12", HEADER + b11.PRE + b11.MOUNT + body)
    b11.write("connectome_sweep12_rclone", HEADER + b11.RCLONE_HEADER + b11.PRE + rclone + body)


if __name__ == "__main__":
    main()
