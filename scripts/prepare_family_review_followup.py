"""Append the one prespecified ViT/SVHN follow-up to the SegFormer retry batch."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts.review_families import OUT, KNOBS, dump
from mlaas_data_generator.cli.manifest.review import save_review_manifest
from mlaas_data_generator.cli.run_manifest import _resolve_row, _validate_row
from mlaas_data_generator.family_preflight import manifest_family_error


def main():
    rows=json.loads((OUT/'followup_expanded.json').read_text())
    mappings=json.loads((OUT/'followup_mapping.json').read_text())
    originals=json.loads((OUT/'pilot_expanded.json').read_text())
    lookup={r['service_id']:r for r in json.loads((OUT/'pilot_mapping.json').read_text())}
    for original in originals:
        if not original['service_id'].startswith('review_78256bd138d4__'):continue
        row=copy.deepcopy(original)
        row.update(service_id=row['service_id'].replace('__pilot_','__followup_'),
                   sample_size=730,requested_quota=730,max_client_quantity=730,
                   unconstrained_sample_size=730,training_epochs=6)
        if any(r['service_id']==row['service_id'] for r in rows):continue
        resolved=_resolve_row(pd.Series(row),{})
        check=_validate_row(resolved)
        error=manifest_family_error(resolved)
        if not check.ok or error:raise ValueError(f'{check}: {error}')
        mapping=copy.deepcopy(lookup[original['service_id']])
        mapping.update(service_id=row['service_id'],previous_pilot_id=original['service_id'],
            requested_knobs={k:row.get(k) for k in KNOBS},
            intervention='All initial 256-image, 96-update trials weak (accuracy 0.192-0.214). One bounded data/training-opportunity follow-up: 730 IID images, 6 epochs; same initialization, evaluation, seeds and remaining recipe knobs.')
        rows.append(row);mappings.append(mapping)
    dump(OUT/'followup_expanded.json',rows)
    dump(OUT/'followup_mapping.json',mappings)
    save_review_manifest(pd.DataFrame(rows),OUT/'followup.csv')
    print(f'Prepared {len(rows)} follow-up services')


if __name__=='__main__':main()
