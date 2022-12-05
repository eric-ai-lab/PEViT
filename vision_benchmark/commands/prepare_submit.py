"""
submit predictions to leaderboard service
"""
import argparse
from collections import defaultdict
import json
import logging
import pathlib
import zipfile
import itertools
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Submit predictions to leaderboard service.')
    parser.add_argument('--combine_path', required=True, help='Prediction json file path.', type=pathlib.Path)
    parser.add_argument('--combine_name', default='all_predictions', required=False, help='Output file name.', type=str)
    args = parser.parse_args()

    return args


# if you find the accuracy is not enough, pleae consider increasing `prec`.
def json_prec_dump(data, prec=6):
    return json.dumps(json.loads(json.dumps(data), parse_float=lambda x: round(float(x), prec)))


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    all_predictions = defaultdict(list)
    for prediction_file in args.combine_path.iterdir():
        if prediction_file.suffix != '.json':
            print(f'Ignoring file {prediction_file.name} by suffix.')
            continue
        prediction_data = json.loads(prediction_file.read_text())
        all_predictions[prediction_data['dataset_name']].append(prediction_data)

    all_combine_predictions = []

    KNOWN_AVERAGE_KEYS = ['num_trainable_params']
    KNOWN_MERGE_KEYS = ['rnd_seeds', 'predictions']
    KNOWN_DIFF_KEYS = KNOWN_AVERAGE_KEYS + KNOWN_MERGE_KEYS

    for ds, prediction_data in all_predictions.items():
        prediction_keys = list(prediction_data[0])
        combined_dict = dict()
        for key in prediction_keys:
            values = [x[key] for x in prediction_data]
            if key not in KNOWN_DIFF_KEYS:
                assert all(x == values[0] for x in values)
                values = values[0]
            else:
                if key in KNOWN_MERGE_KEYS:
                    values = list(itertools.chain.from_iterable(values))
                elif key in KNOWN_AVERAGE_KEYS:
                    values = np.asarray(values).mean()
                else:
                    assert False
            combined_dict[key] = values
        all_combine_predictions.append(combined_dict)

    all_predictions = {"data": all_combine_predictions}
    all_predictions = json_prec_dump(all_predictions)
    save_path = args.combine_path / f'{args.combine_name}.zip'
    zf = zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED)
    zf.writestr('all_predictions.json', all_predictions)
    zf.close()


if __name__ == '__main__':
    main()
