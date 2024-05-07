import argparse as ag
import json

def parse_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Change Detection')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata
