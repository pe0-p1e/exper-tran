"""Linux physical-memory containment. Never use RLIMIT_AS for CUDA address space."""
import argparse,json,os,subprocess,sys
from pathlib import Path

GIB=2**30
BASE=Path('/sys/fs/cgroup/v8-joint-selection')

def proc_rss_gib(pid):
    try:
        values=dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        return int(values['VmRSS'].split()[0])*1024/GIB
    except (FileNotFoundError,ProcessLookupError,KeyError):return 0.

def snapshot(group):
    p=Path(group)
    return dict(path=str(p),current_gib=int((p/'memory.current').read_text())/GIB,
        peak_gib=int((p/'memory.peak').read_text())/GIB,
        max_gib=int((p/'memory.max').read_text())/GIB,
        events={k:int(v) for k,v in (line.split() for line in (p/'memory.events').read_text().splitlines())})

def attach(pid,start,cap_gib):
    """Privileged helper confines only verified current-user V8 science workers."""
    if os.geteuid()!=0:raise RuntimeError('cgroup setup requires existing sudo authorization')
    proc=Path('/proc')/str(pid);owner=proc.stat().st_uid
    if owner!=int(os.environ.get('SUDO_UID',owner)):raise RuntimeError('Foreign worker owner')
    fields=(proc/'stat').read_text().split()
    if fields[21]!=str(start):raise RuntimeError('PID was reused')
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(Path(__file__).with_name('worker.py')) not in command:raise RuntimeError('Not this study worker')
    if not .0625<=cap_gib<=112:raise ValueError('Invalid physical memory cap')
    BASE.mkdir(exist_ok=True)
    total=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:')))*1024
    parent_cap=int(total-max(12*GIB,total*.1))
    (BASE/'memory.max').write_text(str(parent_cap))
    (BASE/'memory.high').write_text(str(int(parent_cap*.95)))
    (BASE/'memory.swap.max').write_text('0')
    (BASE/'cgroup.subtree_control').write_text('+memory')
    group=BASE/f'worker-{pid}-{start}';group.mkdir(exist_ok=True)
    maximum=int(cap_gib*GIB)
    (group/'memory.max').write_text(str(maximum))
    (group/'memory.high').write_text(str(int(maximum*.95)))
    (group/'memory.swap.max').write_text('0')
    (group/'memory.oom.group').write_text('1')
    (group/'cgroup.procs').write_text(str(pid))
    return snapshot(group)

def enforce(cap_gib):
    pid=os.getpid();start=Path(f'/proc/{pid}/stat').read_text().split()[21]
    p=subprocess.run(['sudo','-n','/usr/bin/python3',str(Path(__file__).resolve()),'--attach',str(pid),
        '--start',start,'--cap-gib',str(cap_gib)],text=True,capture_output=True,check=True)
    result=json.loads(p.stdout)
    actual=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
    if str(actual)!=result['path']:raise RuntimeError('Worker cgroup placement mismatch')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--attach',type=int,required=True);ap.add_argument('--start',required=True);ap.add_argument('--cap-gib',type=float,required=True)
    a=ap.parse_args();print(json.dumps(attach(a.attach,a.start,a.cap_gib)))
