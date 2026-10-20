"""Bounded, read-only call-graph exploration using angr (not Ghidra).

Owner-supplied PE/ELF/Mach-O binary is supplied locally; the tool does NOT
execute the target, write a database, or copy target bytes into this repo.

Usage:
    pip install angr
    python independent_angr_callgraph.py /path/to/binary \
        --entry 0x1402d4e80 --entry 0x1402d4600 --max-blocks 140

The traversal is intentionally bounded; callees are observed direct branches
only. Indirect calls and code outside the searched window remain unresolved.
"""
import argparse
import json
from collections import Counter, deque
import angr


def scan(project, root, max_blocks=140, window=0x1800):
    queue=deque([root])
    visited=set()
    calls=Counter()
    ops=Counter()
    errors=[]
    while queue and len(visited)<max_blocks:
        addr=queue.popleft()
        if addr in visited: continue
        visited.add(addr)
        try:
            b=project.factory.block(addr,opt_level=0,num_inst=100)
            instructions=b.capstone.insns
            if not instructions: continue
            ops.update(i.mnemonic for i in instructions)
            last=instructions[-1]
            next_addr=last.address+last.size
            if last.mnemonic.startswith("call"):
                if last.op_str.startswith("0x"):
                    calls[int(last.op_str,16)]+=1
                if next_addr<root+window:
                    queue.append(next_addr)
            elif last.mnemonic.startswith("j"):
                if last.op_str.startswith("0x"):
                    target=int(last.op_str,16)
                    if abs(target-root)<window:
                        queue.append(target)
                if last.mnemonic!="jmp" and next_addr<root+window:
                    queue.append(next_addr)
            elif next_addr<root+window:
                queue.append(next_addr)
        except Exception as exc:
            errors.append({"address":hex(addr),"reason":str(exc)[:220]})
    return {
        "entry":hex(root),
        "basic_blocks_visited":len(visited),
        "bounds":{"max_blocks":max_blocks,"region_window_bytes":window},
        "errors":errors[:8],
        "floating_point_instructions_approx":sum(
            n for op,n in ops.items() if any(k in op for k in ("sd","ss","pd","ps"))
        ),
        "observed_direct_callees":[{"address":hex(a),"calls":n}
                                   for a,n in calls.most_common(50)],
        "limitations":"A bounded disassembly is not a complete function CFG; "
                      "unresolved indirect jumps and data-dependent calls omitted."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary")
    p.add_argument("--entry",action="append",type=lambda s:int(s,0),required=True)
    p.add_argument("--max-blocks",type=int,default=140)
    p.add_argument("--window",type=lambda s:int(s,0),default=0x1800)
    a=p.parse_args()
    if a.max_blocks<1 or a.window<1:
        p.error("max-blocks/window must be positive")
    project=angr.Project(a.binary,auto_load_libs=False)
    print(json.dumps({
        "analysis_engine":"angr",
        "engine_version":angr.__version__,
        "arch":str(project.arch),
        "calls":[scan(project,addr,a.max_blocks,a.window) for addr in a.entry]
    },indent=2))


if __name__=="__main__":
    main()
