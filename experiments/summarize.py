#!/usr/bin/env python3
"""Audit published evidence and rebuild CSVs without models, CUDA or third-party packages.

python experiments/summarize.py         # verify then write derived CSVs
python experiments/summarize.py --check # also check existing CSVs; no writes

Omitted binary arrays/rotations are checked against recorded hashes, not re-read.
The full local-archive audit is preserved separately in checks/full_archive_audit.json.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def require(passed, message):
    if not passed:
        raise ValueError(message)


def render_csv(rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    return stream.getvalue().encode('utf-8-sig')


def summarize(check_only=False):
    manifest = read('publication_manifest.json')
    for name, record in manifest['files'].items():
        path = ROOT / name
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'],
                'Published evidence changed or missing: ' + name)
    protocol, status = read('protocol.json'), read('status.json')
    require(protocol['plan'] == 'A' and protocol['seeds'] == [0], 'Unexpected protocol')
    require(status['state'] == 'complete' and not status['failed'] and not status['skipped'], 'Incomplete experiment')
    require(len(status['completed']) == len(protocol['tasks']) == 32, 'Task count differs')
    source_hashes = read('source_sha256.json')
    for name, record in manifest['files'].items():
        if name.startswith('source/'):
            require(source_hashes[name.removeprefix('source/')] == record['sha256'], 'Frozen source mismatch: ' + name)
    shared, original_files = read('shared_sha256.json'), read('download_sha256.json')
    evals, trains, losses, performance = [], [], [], []
    for task in protocol['tasks']:
        tid = task['id']
        result, exitinfo = read(f'results/{tid}.json'), read(f'logs/{tid}.exit.json')
        log = (ROOT / f'logs/{tid}.log').read_text()
        finals = [json.loads(line[len('FINAL_RESULT '):]) for line in log.splitlines()
                  if line.startswith('FINAL_RESULT ')]
        require(finals == [result] and result['status'] == 'complete' and exitinfo['exit_code'] == 0,
                'Log/result/exit mismatch: ' + tid)
        require('Traceback (most recent call last)' not in log, 'Traceback: ' + tid)
        require(all(result.get(k) == v for k, v in task.items()), 'Task metadata mismatch: ' + tid)
        performance.append(dict(id=tid, model=result['model'], stack=result['stack'], kind=result['kind'],
            method=result.get('method', 'W16A4KV4 training'), rotate=result.get('rotate'), akv_bits=result.get('bits'),
            wall_seconds=exitinfo['wall_seconds'], worker_seconds=result['worker_seconds'],
            model_load_seconds=result['model_load_seconds'], training_seconds=result.get('training_seconds'),
            quant_seconds=result.get('quant_seconds'), warmup_seconds=result.get('warmup_seconds'),
            eval_seconds=result.get('eval_seconds'), evaluated_tokens_per_second=result.get('evaluated_tokens_per_second'),
            max_rss_gib=result['max_rss_kib'] / 1024**2,
            sampled_peak_gpu_mib=exitinfo['sampled_peak_gpu_memory_mib'],
            result_file=f'results/{tid}.json', log_file=f'logs/{tid}.log',
            exit_file=f'logs/{tid}.exit.json'))
        if result['kind'] == 'train':
            steps = [json.loads(line[len('TRAIN_STEP '):]) for line in log.splitlines()
                     if line.startswith('TRAIN_STEP ')]
            with (ROOT / f'training/{tid}/loss.csv').open() as handle:
                saved = list(csv.DictReader(handle))
            require(len(steps) == len(saved) == result['steps'] == result['loss_records'] == 100, 'Loss count: ' + tid)
            require([s['step'] for s in steps] == list(range(100)), 'Step sequence: ' + tid)
            for s, c in zip(steps, saved):
                require(all(math.isfinite(v) and float(c[k]) == v for k, v in s.items()), 'Loss CSV differs: ' + tid)
                require(abs(s['learning_rate'] - .75 * (1 + math.cos(math.pi * s['step'] / 100))) < 1e-12,
                        'Cosine schedule differs: ' + tid)
                losses.append(dict(model=result['model'], stack=result['stack'], seed=result['seed'], **s))
            require(steps[0]['loss'] == result['first_loss'] and steps[-1]['loss'] == result['last_loss'] and
                    abs(statistics.mean(s['loss'] for s in steps) - result['mean_loss']) < 1e-12, 'Loss statistics: ' + tid)
            require(result['rotation_sha256'] == original_files[f'rotations/{tid}/R.npy'], 'Recorded rotation hash: ' + tid)
            require(result['training_data_sha256'] == shared[f"{result['model']}_train_s0.npy"] and
                    result['initial_rotation_sha256'] == shared[f"{result['model']}_init_s0.npy"], 'Training input hash: ' + tid)
            trains.append(dict(result, first_lr=steps[0]['learning_rate'], last_lr=steps[-1]['learning_rate'],
                first10_mean_loss=statistics.mean(s['loss'] for s in steps[:10]),
                last10_mean_loss=statistics.mean(s['loss'] for s in steps[-10:]),
                probe_relative_change=result['final_fixed_probe_loss'] / result['initial_fixed_probe_loss'] - 1,
                log_file=f'logs/{tid}.log', result_file=f'results/{tid}.json'))
        else:
            require(math.isfinite(result['ppl']) and abs(math.exp(result['nll_sum'] / result['evaluated_tokens']) - result['ppl']) < 1e-9,
                    'PPL definition mismatch: ' + tid)
            require(result['test_sha256'] == shared[f"{result['model']}_test.npy"], 'Test hash: ' + tid)
            blocks = 166 if result['model'] == 'tinyllama' else 163
            require(result['test_blocks'] == blocks and result['evaluated_tokens'] == blocks * 2047, 'Token count: ' + tid)
            if result['rotate']:
                ext = 'bin' if result['stack'] == 'official' else 'npy'
                rotation = f"rotations/{result['model']}_{result['stack']}_train_s0/R.{ext}"
                require(result['rotation_sha256'] == original_files[rotation], 'Evaluation rotation hash: ' + tid)
            if result['method'] == 'gptq':
                require(result['calibration_sha256'] == shared['tinyllama_gptq_s0.npy'], 'GPTQ calibration hash: ' + tid)
                if result['stack'] == 'jittor':
                    coverage = set(re.findall(r'GPTQ layer (\d+) ([\w.]+): (\d+) batches / (\d+) tokens', log))
                    require(len(coverage) == 154 and all(h[2:] == ('128', '262144') for h in coverage), 'GPTQ coverage: ' + tid)
            evals.append(dict(result, log_file=f'logs/{tid}.log', result_file=f'results/{tid}.json'))
    require((len(trains), len(evals), len(losses), len(performance)) == (4, 28, 400, 32), 'Summary counts')
    require(read('summary_status.json') == dict(completed_train=4, completed_eval=28, required_train=4, required_eval=28, missing=[]),
            'Raw summary status differs')
    alignment, metrics, gains = [], [], []
    for model in ('tinyllama', 'mistral'):
        pair = [t for t in trains if t['model'] == model]
        require(pair[0]['training_data_sha256'] == pair[1]['training_data_sha256'] and
                pair[0]['initial_rotation_sha256'] == pair[1]['initial_rotation_sha256'] and
                pair[0]['base_weight_dtype'] == pair[1]['base_weight_dtype'], 'Training pairing: ' + model)
        x, y = [[s['loss'] for s in losses if s['model'] == model and s['stack'] == stack] for stack in ('official', 'jittor')]
        mx, my = statistics.mean(x), statistics.mean(y)
        metrics.append(dict(model=model, loss_mae=statistics.mean(abs(a-b) for a,b in zip(x,y)),
            loss_rmse=math.sqrt(statistics.mean((a-b)**2 for a,b in zip(x,y))),
            loss_pearson_r=sum((a-mx)*(b-my) for a,b in zip(x,y)) /
                math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))))
        for a in [r for r in evals if r['model'] == model and r['stack'] == 'official']:
            b = next(r for r in evals if r['model'] == model and r['stack'] == 'jittor' and
                     (r['method'], r['bits'], r['rotate']) == (a['method'], a['bits'], a['rotate']))
            require(a['test_sha256'] == b['test_sha256'] and a['evaluated_tokens'] == b['evaluated_tokens'], 'Paired PPL input: ' + a['id'])
            alignment.append(dict(model=model, method=('SpinQuant+' if a['rotate'] else '') + a['method'].upper(),
                precision=('FP32' if model == 'tinyllama' else 'FP16') if a['method'] == 'float' else ('W4A16' if a['bits'] == 16 else 'W4A4KV4'),
                official_ppl=a['ppl'], jittor_ppl=b['ppl'], signed_delta=b['ppl']-a['ppl'],
                signed_relative_pct=100*(b['ppl']/a['ppl']-1), absolute_relative_pct=100*abs(b['ppl']/a['ppl']-1),
                official_id=a['id'], jittor_id=b['id']))
        for base in [r for r in evals if r['model'] == model and not r['rotate'] and r['method'] != 'float']:
            spin = next(r for r in evals if (r['model'], r['stack'], r['method'], r['bits'], r['rotate']) ==
                        (model, base['stack'], base['method'], base['bits'], True))
            gains.append(dict(model=model, stack=base['stack'], method=base['method'].upper(),
                precision='W4A16' if base['bits'] == 16 else 'W4A4KV4', direct_ppl=base['ppl'],
                spinquant_ppl=spin['ppl'], ppl_reduction_pct=100*(1-spin['ppl']/base['ppl'])))
    require(len(alignment) == 14 and len(gains) == 12, 'Pair/gain counts')
    # Compare independent reconstruction to the archived, full-array analysis.
    reference = read('checks/original_analysis_data.json')
    for row in alignment:
        match = next(r for r in reference['alignment'] if r['official_id'] == row['official_id'])
        require(row == match, 'Original PPL analysis differs: ' + row['official_id'])
    for r in performance:
        old = next(x for x in reference['performance'] if x['id'] == r['id'])
        require(all(r[k] == v for k, v in old.items() if k != 'source'), 'Original performance differs: ' + r['id'])
    tables = {'results.csv': evals, 'alignment.csv': alignment, 'training_loss.csv': losses,
              'training_summary.csv': trains, 'performance.csv': performance,
              'rotation_gains.csv': gains, 'loss_alignment_metrics.csv': metrics}
    for name, rows in tables.items():
        content = render_csv(rows)
        if check_only:
            require((ROOT / name).read_bytes() == content, 'Derived table stale: ' + name)
        else:
            (ROOT / name).write_bytes(content)
    report = dict(state='passed', published_evidence_files_verified=len(manifest['files']),
                  train=4, eval=28, paired_eval=14, loss_records=400, performance_records=32,
                  scope='Published logs/source bytes, result consistency and recorded omitted-input hashes. No model execution or omitted binary revalidation.')
    if not check_only:
        (ROOT / 'evidence_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    summarize(parser.parse_args().check)
