"""Brute-force verifier for bit_manipulation puzzles.

Searches a constrained-but-expressive grammar of 8-bit transforms to find a rule
consistent with ALL example pairs, then predicts the query output. Used to:
  (a) cross-check that the dataset ground-truth answer is consistent with examples,
  (b) confirm whether a captured trace's logic could be sound.

Grammar of "terms" (each an 8-bit value derived from input x):
  - x (identity)
  - ROL k / ROR k (rotate, k=1..7)
  - SHL k / SHR k (logical shift, k=1..7)
  - NOT x
  - reverse(x) (bit reversal)
  - neighbor XORs (rule-90/150 style cyclic): nb_lr = ROL1 ^ ROR1, etc.
Then we try output = combine(termA, op, termB) [^ optional NOT], plus single-term.
op in {AND, OR, XOR, ANDN, ORN}. Also try output = termA directly.
This is heuristic: not exhaustive over all 2^(8*256) functions, but covers the
puzzle's stated operation classes (shift/rotate/XOR/AND/OR/NOT/majority/choice).
"""
import json, sys, re
from itertools import product

M = 0xFF

def rol(x, k): k%=8; return ((x<<k)|(x>>(8-k)))&M if k else x
def ror(x, k): k%=8; return ((x>>k)|(x<<(8-k)))&M if k else x
def shl(x, k): return (x<<k)&M
def shr(x, k): return (x>>k)&M
def notx(x): return (~x)&M
def rev(x):
    r=0
    for i in range(8):
        r=(r<<1)|((x>>i)&1)
    return r&M

def terms(x):
    t={'x':x,'NOT':notx(x),'REV':rev(x)}
    for k in range(1,8):
        t[f'ROL{k}']=rol(x,k); t[f'ROR{k}']=ror(x,k)
        t[f'SHL{k}']=shl(x,k); t[f'SHR{k}']=shr(x,k)
    # majority / CA neighbor terms
    l=rol(x,1); r=ror(x,1)
    t['NB_XOR']=(l^r)&M          # rule90 cyclic
    t['NB_XOR3']=(l^x^r)&M       # rule150 cyclic
    t['NB_MAJ']=((l&r)|(l&x)|(r&x))&M  # majority of 3 cyclic
    return t

TERMNAMES=list(terms(0).keys())

def gen_rules():
    # single term, optional NOT
    for tn in TERMNAMES:
        yield ('S', tn, False)
        yield ('S', tn, True)
    # term op term, optional NOT, optional xor-const
    ops=['AND','OR','XOR']
    for a in TERMNAMES:
        for b in TERMNAMES:
            for op in ops:
                yield ('B', a, b, op, False)
                yield ('B', a, b, op, True)

def apply_rule(rule, x):
    t=terms(x)
    if rule[0]=='S':
        _,tn,neg=rule
        v=t[tn]
        return notx(v) if neg else v
    else:
        _,a,b,op,neg=rule
        va,vb=t[a],t[b]
        if op=='AND': v=va&vb
        elif op=='OR': v=va|vb
        else: v=va^vb
        v&=M
        return notx(v) if neg else v

def find_rules(examples):
    """Return list of rules consistent with all examples."""
    good=[]
    for rule in gen_rules():
        if all(apply_rule(rule, xi)==yi for xi,yi in examples):
            good.append(rule)
    return good

def parse_prompt(prompt):
    ex=re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    examples=[(int(a,2),int(b,2)) for a,b in ex]
    qm=re.search(r'output for:\s*([01]{8})', prompt)
    query=int(qm.group(1),2) if qm else None
    return examples, query

def b8(v): return format(v,'08b')

if __name__=='__main__':
    fn='puzzle_team/data/bit_manipulation.jsonl'
    recs=[json.loads(l) for l in open(fn)]
    n=len(recs)
    consistent=0; pred_match=0; no_rule=0; ambiguous=0; mismatch=[]
    import random
    random.seed(0)
    sample=recs if len(sys.argv)>1 and sys.argv[1]=='all' else random.sample(recs,200)
    for r in sample:
        examples,query=parse_prompt(r['prompt'])
        gt=r['answer']
        rules=find_rules(examples)
        if not rules:
            no_rule+=1
            continue
        consistent+=1
        preds=set(b8(apply_rule(ru,query)) for ru in rules)
        if len(preds)>1: ambiguous+=1
        if gt in preds:
            pred_match+=1
        else:
            mismatch.append((r['id'],gt,sorted(preds)[:3]))
    print(f'sample={len(sample)} rule_found={consistent} no_rule_in_grammar={no_rule}')
    print(f'  of rule_found: gt_consistent_with_predicted={pred_match}  ambiguous_multi_pred={ambiguous}')
    print(f'  mismatches (gt not in grammar prediction): {len(mismatch)}')
    for m in mismatch[:10]: print('   ',m)
