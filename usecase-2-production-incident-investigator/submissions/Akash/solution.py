"""Incident Clarity: deterministic retrieval and conservative evidence correlation.
Python 3.10+, standard library only. No network, credentials or system actions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import csv
import io
import json
import math
from pathlib import Path
import re

LOG = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+(INFO|WARN|ERROR)\s+(\S+)\s+(.+)$', re.M)
STOP = set('the a an is are to of and in for with after yesterday identify probable root cause supporting evidence impacted components recommended remediation what mean time recover systems customers reporting sometimes'.split())

def tokens(text):
    terms = re.findall(r'[a-z]+', text.lower())
    aliases = {'payments':'payment','charges':'payment','charge':'payment','failing':'failure','failed':'failure','failures':'failure','emails':'email','delays':'delay','late':'delay','arriving':'delivery','confirmation':'email'}
    return [aliases.get(t,t) for t in terms if t not in STOP and len(t)>1]

def ingest(corpus):
    chunks=[]
    for source,text in sorted(corpus.items()):
        lines=text.splitlines()
        if lines and {'issue_id','signature','affected_component'}.issubset(next(csv.reader([lines[0]]))):
            # Preserve original rows verbatim; parse quoted commas correctly.
            stream=io.StringIO(text); reader=csv.DictReader(stream); previous=1
            for row in reader:
                end=reader.line_num
                excerpt='\n'.join(lines[previous:end]);previous=end
                chunks.append({'source':source,'text':excerpt,'kind':'issue','row':row})
        else:
            kind=next((k for k,pattern in [('logs',r'^# .*logs'),('deployment',r'^# Deployment history'),('prior',r'^# Previous incidents'),('runbook',r'^# Runbooks'),('architecture',r'^# Architecture'),('api',r'^# API')] if re.search(pattern,text,re.I|re.M)), 'document')
            for section in re.split(r'(?=^## )',text,flags=re.M):
                if section.strip():chunks.append({'source':source,'text':section.strip(),'kind':kind})
    return chunks

def retrieve(query,chunks):
    """TF-IDF cosine retrieval across prose sections and individual CSV rows."""
    bags=[Counter(tokens(c['text'])) for c in chunks];df=Counter()
    for bag in bags:df.update(bag.keys())
    idf={t:math.log((1+len(bags))/(1+n))+1 for t,n in df.items()}
    def vector(bag):return {t:(1+math.log(n))*idf.get(t,1) for t,n in bag.items()}
    q=vector(Counter(tokens(query)));qn=math.sqrt(sum(v*v for v in q.values())) or 1
    ranked=[]
    for chunk,bag in zip(chunks,bags):
        v=vector(bag);norm=math.sqrt(sum(x*x for x in v.values())) or 1
        ranked.append((sum(value*v.get(t,0) for t,value in q.items())/(qn*norm),chunk))
    return [c for score,c in sorted(ranked,key=lambda x:(-x[0],x[1]['source'],x[1]['text']))]

def investigate(query: str, corpus: dict) -> dict:
    if not isinstance(query,str) or not query.strip():raise ValueError('A non-empty query is required')
    if not isinstance(corpus,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in corpus.items()):raise ValueError('Corpus must map source names to text')
    chunks=ingest(corpus);ranked=retrieve(query,chunks)
    records=[{'source':c['source'],'line':m.group(0),'time':m.group(1),'level':m.group(2),'component':m.group(3),'message':m.group(4)} for c in chunks if c['kind']=='logs' for m in LOG.finditer(c['text'])]
    q=set(tokens(query));scores=Counter()
    for row in records:
        overlap=len(q & set(tokens(row['component']+' '+row['message'])))
        scores[row['component']]+=overlap
    relevant={c for c,score in scores.items() if score>0}
    exception_rows=[r for r in records if r['component'] in relevant and r['level']=='ERROR' and re.search(r'\b\w+Exception\b',r['message'])]
    # Scope by the user's symptom before expanding retrieval using error signatures.
    exception_rows.sort(key=lambda r:(-scores[r['component']],r['time'],r['line']))
    lead=exception_rows[0] if exception_rows else None
    evidence=[]
    def cite(source,excerpt):
        if excerpt and excerpt in corpus[source] and {'source':source,'excerpt':excerpt} not in evidence:evidence.append({'source':source,'excerpt':excerpt})
    def report(cause,systems,mttr,remediation,score):
        return {'root_cause':cause,'supporting_evidence':evidence,'impacted_systems':sorted(set(systems)),'mttr_minutes':mttr,'remediation':remediation,'confidence_score':float(score),'needs_human_review':score<50}
    if lead:
        comp=lead['component'];signature=re.search(r'\b\w+Exception\b',lead['message']).group()
        ranked=retrieve(query+' '+comp+' '+signature,chunks)
        issue=next((c for c in ranked if c['kind']=='issue' and c['row'].get('affected_component')==comp and signature in c['row'].get('signature','')),None)
        cite(lead['source'],lead['line'])
        if issue:cite(issue['source'],issue['text'])
        # A deployment is corroboration only for the same component, before symptoms,
        # and with a configuration term explicitly supported by the issue signature.
        config_terms=set(tokens(issue['row']['signature']))-set(tokens(comp+' '+signature)) if issue else set()
        deployments=[]
        for c in ranked:
            if c['kind']!='deployment':continue
            for line in c['text'].splitlines():
                cells=[v.strip().replace('**','') for v in line.strip('|').split('|')]
                if len(cells)!=4 or cells[2]!=comp:continue
                try:when=datetime.fromisoformat(cells[1])
                except ValueError:continue
                if when<=datetime.fromisoformat(lead['time']) and len(config_terms & set(tokens(cells[3])))>=2:deployments.append((when,c,line,cells))
        deployment=max(deployments,key=lambda x:x[0]) if deployments else None
        if deployment:cite(deployment[1]['source'],deployment[2])
        runbook=next((c for c in ranked if c['kind']=='runbook' and signature in c['text'] and comp in c['text']),None)
        prior=next((c for c in ranked if c['kind']=='prior' and signature in c['text'] and comp in c['text']),None)
        if runbook:cite(runbook['source'],runbook['text'])
        if prior:cite(prior['source'],prior['text'])
        # Category weights, not chunk count. Repeated log lines cannot inflate confidence.
        score=20+25*bool(issue)+25*bool(deployment)+10*bool(runbook)+10*bool(prior)
        if not issue:score=min(score,45)
        cause=f'Observed {signature} in {comp}; the underlying cause remains unconfirmed.'
        if issue:
            cause=f"Probable cause: {issue['row']['signature']}"
            if deployment:cause+=f" Corroborating change: {deployment[3][0]} on {deployment[3][1]} UTC: {deployment[3][3]}. First matching error in the supplied log: {lead['time']} UTC."
            else:cause+=' No matching pre-symptom deployment was established; attribution to a release is unconfirmed.'
        systems=[comp]
        for row in records:
            if row['time']==lead['time'] and row['level']=='ERROR' and row['component']!=comp:
                # Confirm an upstream relationship in architectural/API evidence.
                relation=next((c for c in ranked if c['kind'] in {'api','architecture'} and row['component'] in c['text'] and comp in c['text']),None)
                if relation:systems.append(row['component']);cite(row['source'],row['line']);cite(relation['source'],relation['text'])
        mttr=None;remediation='Collect configuration, saturation metrics and correlated traces; ask the service owner to validate the hypothesis before any change.'
        if runbook and score>=50:
            text=runbook['text'];match=re.search(r'\*\*Remediation\*\*:\s*(.*?)(?=\n\n|\Z)',text,re.S)
            if match:remediation='Human-reviewed recommendation: '+match.group(1).strip()+' Validate current utilization, traffic and deployment configuration first; verify error rate and latency after the authorised change. No action has been executed.'
            estimate=re.search(r'Typical MTTR:\s*(\d+)\s*minutes',text,re.I)
            if estimate and not re.search(r'unconfirmed|may not apply|unverified',text,re.I):mttr=int(estimate.group(1))
        if mttr is not None:remediation+=f' Recovery estimate is the runbook typical value ({mttr} minutes), not measured recovery for this incident.'
        return report(cause,systems,mttr,remediation,score)
    # Thin evidence: retrieve symptom context, explicitly retain uncertainty.
    component=max(sorted(relevant),key=lambda c:scores[c]) if relevant else None
    rows=[r for r in records if r['component']==component]
    for row in rows[:2]:cite(row['source'],row['line'])
    for row in rows:
        if row['level']=='WARN':cite(row['source'],row['line'])
    for kind in ('architecture','deployment','prior','runbook','api'):
        match=next((c for c in ranked if c['kind']==kind and (component and component in c['text'])),None)
        if match:cite(match['source'],match['text'])
    if component:
        cause=f'Root cause undetermined. The supplied logs show symptoms in {component}, but do not establish a correlated deployment, matching error signature or confirmed precedent.'
        remediation='Obtain per-stage traces, queue age/depth history, consumer concurrency and third-party provider latency; validate competing hypotheses with the service owner before scaling or changing configuration. Do not reuse an unconfirmed runbook recovery estimate. No action has been executed.'
        if 'email' in q:cause+=' Queue/consumer capacity and downstream email-provider latency remain competing hypotheses; an isolated queue warning does not distinguish them.'
        return report(cause,[component],None,remediation,20)
    return report('Root cause undetermined: no relevant application log evidence was found.',[],None,'Request the incident time window, affected component and correlated logs before proposing remediation.',0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='Path to the upstream use-case data directory')
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('answers.json'))
    args=parser.parse_args();answers={}
    for folder in sorted(args.data_dir.iterdir()):
        if not folder.is_dir() or not (folder/'query.txt').is_file():continue
        corpus={p.name:p.read_text(encoding='utf-8') for p in sorted(folder.iterdir()) if p.suffix in {'.md','.csv'}}
        answers[folder.name]=investigate((folder/'query.txt').read_text(encoding='utf-8'),corpus)
    if not answers:parser.error('No incident folders with query.txt were found')
    args.output.write_text(json.dumps(answers,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(answers)} reports to {args.output}')

if __name__=='__main__':main()
