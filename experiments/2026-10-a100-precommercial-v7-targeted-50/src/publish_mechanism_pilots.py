"""Archive compact DI and mechanism pilot records, never images or model caches."""
import argparse
import time
import pp4_publish as publisher
from pp4_protocol import OUT

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    for folder in ['pull_push_di_removal','pull_push_mechanism_pilots']:
        root=OUT/folder
        if args.watch:
            while not (root/'completion.json').exists():time.sleep(15)
        publisher.ROOT=root;publisher.DEST_FOLDER=folder;publisher.publish()
