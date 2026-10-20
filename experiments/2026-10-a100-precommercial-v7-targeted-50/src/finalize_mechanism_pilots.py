"""Finish CPU validation and publish compact results after the small GPU pilots."""
import time
from analyze_mechanism_pilots import analyze,ROOT
import pp4_publish as publisher

if __name__=='__main__':
    while not (ROOT/'completion.json').exists():time.sleep(15)
    analyze()
    publisher.ROOT=ROOT
    publisher.DEST_FOLDER='pull_push_mechanism_pilots'
    publisher.publish()
