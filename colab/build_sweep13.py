"""Write the sweep 13 one-cell Colab launchers with the current pipeline/ code embedded: colab/connectome_sweep13.ipynb
(Drive mounted in the Colab account) and colab/connectome_sweep13_rclone.ipynb (any Colab account, your Drive via
rclone). Sweep 13 = per-group ensembles of 2 models (WORKSPACE.md "Sweep 13").

    python colab/build_sweep13.py
"""
import glob
import os
import pprint
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sweep11 as b11  # the shared pieces: Drive / rclone setup, data download, pipeline files, train()

HEADER = '''# Alpha Connectome, sweep 13: per-group ensembles (average of 2 models). One A100 run (~1 h incl. the 30-min data
# download), self-contained (the whole pipeline is embedded below).
# Starting point: sweep11_mix.zip, LB 0.66021 (groups 1-3: 10a specialists fine-tuned on all of validation, hedge 0.9;
#   group 4: sweep 10a's ft_film~wise). Sweep 12's recipe tweaks were LB noise (0.66007). Speed: a 2-model ensemble costs
#   ~1.8x a single model, 3 models ~2.6x (Mac test) -> at most 2 models per group (~46 min on the grader if groups 1-3
#   all get 2; 60 min is the limit, the notebook refuses > 50 min).
# 13a new, more different candidates: the 4 sweep 9 teacher members t9_h128_s0..3 (trained on train with other seeds,
#   true targets) fine-tuned on each group's validation half A / half B like sweep 11 (-> runs/sweep13_ft).
# 13b ensemble check (runs/sweep13_ens.json, pipeline/enscheck.py): per group, candidates = sweep 12's recipes
#   (ref = the model in sweep11_mix, e6, u0, u02, lr1), the t9 fine-tunes, sweep 11's shared fine-tune (anc), and models
#   not trained on validation; each scored on the validation half it did not train on. Greedy 2-model ensemble from
#   the current model; kept only if the honest gain (picked on one half, scored on the other) is > 0.0005.
# 13c the kept ensembles with their members trained on all of validation (runs/sweep13_final) -> zips/sweep13_ens.zip,
#   per-group speed test against sweep12_routed (31 min on the grader) -> estimated grader minutes.
# Re-runnable: finished steps are skipped. Keep this tab open while it runs.
'''

PULL = ("PULL = ['sweep9/summary.jsonl', 'sweep9/s9_h128/**', 'cache/t_9.npy', 'cache/mask_train.npy', 'mask_model/split.json',\n"
        "        'cache/tv_s9.npy', *[f'sweep9_teachers/t9_h128_s{k}/best.pt' for k in range(4)],\n"
        "        *[f'sweep10a/{n}/best.pt' for n in ('ft_g1', 'ft_g2', 'ft_g3', 'ft_film', 'ft_all')], 'sweep10a_routed/submission/**',\n"
        "        *[f'sweep11_final/vg{g}_lr2_u06_all/best.pt' for g in (1, 2, 3)], 'sweep11_final/v_lr2_u06_anc_all/best.pt',\n"
        "        'sweep11_ft/v_lr2_u06_anc_*/best.pt', 'sweep12_ft/vg*/best.pt', 'sweep12_final/*/best.pt',\n"
        "        'sweep12_routed/submission/**', 'sweep13_*/**', 'sweep13_ens.json']\n"
        "PUSH = ['sweep13_*/**', 'sweep13_ens.json', 'zips/sweep13*']\n")

SWEEP13 = r'''
# ======== sweep 13
SPEC = [f'{OUT_DIR}/sweep10a/ft_g{g + 1}/best.pt' for g in range(3)]  # sweep 10a group specialists (fine-tune starts)
S11 = [f'{OUT_DIR}/sweep11_final/vg{g + 1}_lr2_u06_all/best.pt' for g in range(3)]  # groups 1-3 of sweep11_mix.zip
ANC_ALL = f'{OUT_DIR}/sweep11_final/v_lr2_u06_anc_all/best.pt'
T9S = [f'{OUT_DIR}/sweep9_teachers/t9_h128_s{k}/best.pt' for k in range(4)]
FT_FILM, FT_ALL = f'{OUT_DIR}/sweep10a/ft_film/best.pt', f'{OUT_DIR}/sweep10a/ft_all/best.pt'
G4_ONNX = f'{OUT_DIR}/sweep10a_routed/submission/model_4.onnx'  # ft_film~wise: group 4 of sweep11_mix.zip
S9_ONNX = f'{OUT_DIR}/sweep10a_routed/submission/model_0.onnx'  # s9_h128: the fallback (unsure group detection)
SW11, SW12, SW12F = f'{OUT_DIR}/sweep11_ft', f'{OUT_DIR}/sweep12_ft', f'{OUT_DIR}/sweep12_final'
SW13, SW13F = f'{OUT_DIR}/sweep13_ft', f'{OUT_DIR}/sweep13_final'
REF_SUB = f'{OUT_DIR}/sweep12_routed/submission'  # same models / speed as sweep11_mix: 31 min on the grader
for need in SPEC + S11 + T9S + [ANC_ALL, FT_FILM, FT_ALL, G4_ONNX, S9_ONNX, f'{REF_SUB}/routes.json']:
    assert os.path.exists(need), f'missing in Drive: {need} (sweeps 9-12)'
TV = '/content/wnn/tv_s9.npy'  # s9_h128's predictions on valid (vg4_light's anchor)
if not os.path.exists(TV):
    if os.path.exists(f'{CACHE}/tv_s9.npy'):
        sh(f'cp {CACHE}/tv_s9.npy {TV}')
    else:
        sh(f'cd /content/wnn && python -m pipeline.teacher --ckpts {S9} --data datasets/valid.parquet --out {TV} && cp {TV} {CACHE}/tv_s9.npy')
FLAGS_V = f'--pool valid --teacher-valid {TV} --evals-per-epoch 0 --skip-final-eval'

def ftv(name, part, g, **kw):  # sweep 11's group fine-tune (in sweep11_mix.zip): group g, validation part A | B | all
    return {'name': f'{name}_{part}', 'init': SPEC[g] if g < 3 else S9, 'group': g, 'lr': 2e-4, 'epochs': 3,
            'sched': 'const', 'select': False, 'swa_from': 2, 'valid_part': part, 'unscored_weight': 0.06, **kw}
R12 = {'ref': {}, 'e6': {'epochs': 6, 'swa_from': 4}, 'u0': {'unscored_weight': 0.0}, 'u02': {'unscored_weight': 0.02},
       'lr1': {'lr': 1e-4}}  # sweep 12's recipes (ref = sweep 11's)
G4R = {'vg4_light': {'lr': 5e-5, 'epochs': 1, 'swa_from': 1, 'distill': 0.5, 'teacher': T9, 'valid_distill': 1.0},
       'vg4_head': {'train_part': 'head', 'lr': 1e-3}}

# ---- 13a: the teacher members t9_h128_s0..3 fine-tuned on each group's validation half A / half B
TS = [ftv(f'vg{g + 1}_t9s{k}', part, g, init=T9S[k]) for part in 'AB' for g in range(3) for k in range(4)]
train(TS, SW13, epochs=3, flags=FLAGS_V)
sync_up()

# ---- 13b: ensemble check
pair = lambda d, nm: {'pair': [f'{d}/{nm}_A/best.pt', f'{d}/{nm}_B/best.pt'], 'alpha': 0.9}
MEM = {}
for g in range(3):
    N = g + 1
    m = {f'vg{N}_{r}': pair(SW12, f'vg{N}_{r}') for r in R12}
    m.update({f'vg{N}_t9s{k}': pair(SW13, f'vg{N}_t9s{k}') for k in range(4)})
    m.update({'shared_anc': pair(SW11, 'v_lr2_u06_anc'), 'spec10a': {'fixed': SPEC[g]}, 's9': {'fixed': S9}})
    MEM[str(N)] = {'current': f'vg{N}_ref', 'members': m}
MEM['4'] = {'current': 'film_wise', 'members': {
    'film_wise': {'wise': [S9, FT_FILM, 0.5]}, 's9': {'fixed': S9}, 'all_wise': {'wise': [S9, FT_ALL, 0.5]},
    **{f't9s{k}': {'fixed': T9S[k]} for k in range(4)}, **{nm: pair(SW12, nm) for nm in G4R}}}
ENS = f'{OUT_DIR}/sweep13_ens.json'
if not os.path.exists(ENS):
    json.dump(MEM, open('/content/wnn/members.json', 'w'), indent=1)
    sh(f'set -o pipefail; cd /content/wnn && python -m pipeline.enscheck score --valid-split {SPLIT} --members members.json '
       f'--max-size 2 --out {ENS} 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
    sync_up()
else:
    print(f'ensemble check already made: {ENS}', flush=True)
ens = json.load(open(ENS))
chosen = {}
print('\nper group: the greedy 2-model ensemble and its honest gain (picked on one half, scored on the other):', flush=True)
for gs, r in ens.items():
    final = r['greedy'][-1]['ensemble']
    keep = len(final) > 1 and r['honest_gain'] > 0.0005
    print(f"  group {gs}: {' + '.join(final)}  pooled {r['greedy'][-1]['score'] - r['current_score']:+.4f}, "
          f"honest {r['honest_gain']:+.4f} -> {'KEEP' if keep else 'no'}", flush=True)
    if keep:
        chosen[int(gs) - 1] = final

# ---- 13c: the kept ensembles, members trained on all of validation -> submission, speed
def final_spec(g, nm):
    """(export spec, training config or None) of member nm of group g, trained on all of validation."""
    N, cfg = g + 1, None
    if nm == f'vg{N}_ref':
        return {'ckpt': S11[g], 'alpha': 0.9}, None
    if nm == 'shared_anc':
        return {'ckpt': ANC_ALL, 'alpha': 0.9}, None
    if nm == 'spec10a':
        return {'ckpt': SPEC[g]}, None
    if nm == 's9':
        return {'ckpt': S9}, None
    if nm == 'film_wise':
        return {'wise': [S9, FT_FILM, 0.5]}, None
    if nm == 'all_wise':
        return {'wise': [S9, FT_ALL, 0.5]}, None
    if g == 3 and nm.startswith('t9s'):
        return {'ckpt': T9S[int(nm[-1])]}, None
    if nm.startswith(f'vg{N}_t9s'):
        cfg = ftv(nm, 'all', g, init=T9S[int(nm[-1])])
    elif nm in G4R:
        cfg = ftv(nm, 'all', 3, **G4R[nm])
    else:
        if os.path.exists(f'{SW12F}/{nm}_all/best.pt'):
            return {'ckpt': f'{SW12F}/{nm}_all/best.pt', 'alpha': 0.9}, None
        cfg = ftv(nm, 'all', g, **R12[nm.split('_', 1)[1]])
    return {'ckpt': f"{SW13F}/{cfg['name']}/best.pt", 'alpha': 0.9}, cfg

if not chosen:
    print('\ndecision: no 2-model ensemble beat the current model by > 0.0005 on held-out halves -> nothing to submit; '
          'sweep11_mix.zip stays the final pick.', flush=True)
else:
    member_specs, todo = {}, []
    for g in sorted(chosen):
        specs = [final_spec(g, nm) for nm in chosen[g]]
        todo += [c for _, c in specs if c]
        member_specs[g] = [s for s, _ in specs]
    if todo:
        train(todo, SW13F, epochs=max(c['epochs'] for c in todo), flags=FLAGS_V)
    def export(groups, sub):
        """Routed submission: the kept ensembles in `groups`, sweep11_mix's model elsewhere."""
        if not os.path.exists(f'{sub}/routes.json'):
            route = [member_specs[g] if g in groups else ([{'ckpt': S11[g], 'alpha': 0.9}] if g < 3 else [{'onnx': G4_ONNX}])
                     for g in range(4)]
            json.dump(route, open('/content/wnn/route.json', 'w'), indent=1)
            sh(f'set -o pipefail; cd /content/wnn && python -m pipeline.enscheck export --route route.json --fallback {S9_ONNX} '
               f'--out {sub} 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
        return sub
    ALL = export(set(chosen), f'{OUT_DIR}/sweep13_ens_all/submission')  # every kept ensemble: timed per group
    sh(f'cd /content/wnn && python -m pipeline.speed --per-group --json /content/wnn/speed.json {ALL} {REF_SUB}')
    sp = json.load(open('/content/wnn/speed.json'))
    # grader minutes ~ 31 (sweep12_routed = sweep11_mix's speed) x time ratio, at the validation and the train group mix;
    # a group without its ensemble runs at sweep11_mix's speed
    MIXES = {'validation': [398, 347, 431, 697], 'train': [1305, 1257, 3488, 4557]}
    def est(groups):
        t = lambda w, sub_of: sum(sp[sub_of(g)][f'g{g + 1}'] * w[g] for g in range(4)) / sum(w)
        return {k: 31.0 * t(w, lambda g: ALL if g in groups else REF_SUB) / t(w, lambda g: REF_SUB) for k, w in MIXES.items()}
    SHARE = [0.28, 0.16, 0.08, 0.48]  # each group's share of the metric weight
    value = lambda groups: sum(SHARE[g] * ens[str(g + 1)]['honest_gain'] for g in groups)
    import itertools
    fits = [set(c) for r in range(len(chosen) + 1) for c in itertools.combinations(sorted(chosen), r) if max(est(c).values()) <= 50]
    keep = max(fits, key=value)
    for g in sorted(chosen):
        print(f"  group {g + 1}: {' + '.join(chosen[g])}, honest gain {ens[str(g + 1)]['honest_gain']:+.4f} "
              f"(x metric share {SHARE[g]:.2f} = {SHARE[g] * ens[str(g + 1)]['honest_gain']:+.5f})"
              + ('' if g in keep else '  -> dropped: does not fit the time budget with the others, or worth least'), flush=True)
    if not keep:
        print('\ndecision: no kept ensemble fits the time budget (50 of 60 min) -> nothing to submit; sweep11_mix.zip stays.', flush=True)
    else:
        SUB = export(keep, f'{OUT_DIR}/sweep13_ens/submission')
        os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
        print(shutil.make_archive(f'{OUT_DIR}/zips/sweep13_ens', 'zip', SUB), flush=True)
        sync_up()
        e = est(keep)
        print(f"\ndecision: zips/sweep13_ens.zip = ensembles in groups {[g + 1 for g in sorted(keep)]}, sweep11_mix's models "
              f"elsewhere. Expected (honest, validation) gain ~ {value(keep):+.5f} pooled; the LB moved ~1-3x the validation "
              f"gain so far.\n  Estimated grader time {e['validation']:.0f} min at the validation group mix, {e['train']:.0f} min "
              f"at the train mix (limit 60; sweep11_mix 31). Submit it (LB to beat: 0.66021).", flush=True)
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
    body = body[:body.index("def table(")] + SWEEP13
    files = {f"pipeline/{os.path.basename(p)}": open(p).read() for p in sorted(glob.glob(os.path.join(b11.ROOT, "pipeline", "*.py")))}
    b11.FILES.clear()
    b11.FILES.update(files)
    body = body.replace("__FILES__", pprint.pformat(files, width=120))
    b11.write("connectome_sweep13", HEADER + b11.PRE + b11.MOUNT + body)
    b11.write("connectome_sweep13_rclone", HEADER + b11.RCLONE_HEADER + b11.PRE + rclone + body)


if __name__ == "__main__":
    main()
