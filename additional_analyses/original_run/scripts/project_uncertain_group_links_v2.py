"""Repair only literal-reference parsing; retain v1 script/output as history."""
import csv,hashlib,io,json,re
from pathlib import Path

PATTERN=re.compile(r'GROUP_REVIEW_UNCERTAIN:(v2_parent_neutralized:(host|guest):[0-9a-f]{16})(?= \| |$)')

def main():
    # The axis is explicit inside each frozen group ID, even when a preceding
    # reason used the only host:/guest: prefix for the entire reason list.
    tests=[
        ('host:GROUP_REVIEW_UNCERTAIN:v2_parent_neutralized:host:60375fd0ab73989e',[('v2_parent_neutralized:host:60375fd0ab73989e','host')]),
        ('host:COUNTERION_REMOVAL_CANDIDATE_NOT_AUTHOR_CONFIRMED | GROUP_REVIEW_UNCERTAIN:v2_parent_neutralized:host:60375fd0ab73989e',[('v2_parent_neutralized:host:60375fd0ab73989e','host')]),
        ('host:UNASSIGNED_STEREOCENTER_POTENTIAL | guest:GROUP_REVIEW_UNCERTAIN:v2_parent_neutralized:guest:1b53e3ca6107e76a',[('v2_parent_neutralized:guest:1b53e3ca6107e76a','guest')]),
    ]
    checked=[]
    for text,expected in tests:
        actual=PATTERN.findall(text)
        if actual!=expected:raise AssertionError((actual,expected))
        checked.append({'actual':actual,'expected':expected})
    root=Path(__file__).resolve().parents[1]
    receipt=json.loads((root/'inputs/projection_receipt.json').read_text('utf-8'))
    source=receipt['sources']['parent_appearance_keys']
    payload=Path(source['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=source['sha256']:
        raise RuntimeError('source drift before corrected identity-only projection')
    links=set()
    for row in csv.DictReader(io.StringIO(payload.decode('utf-8-sig'),newline='')):
        if row['layer']!='v2_reviewed_only_parent':continue
        for group,axis in PATTERN.findall(row['uncertain_reason']):
            links.add((axis,row[axis+'_id'],group))
    output=root/'inputs/identity/uncertain_group_links_v2.csv'
    with output.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(['role','raw_id','review_group_id'])
        writer.writerows(sorted(links,key=lambda r:(r[0],int(r[1]),r[2])))
    result={'source':source,'projection':str(output),'rows':len(links),'group_count':len({r[2] for r in links}),
            'method':'Extract the literal frozen group reference, with axis declared in the reference itself.',
            'repair_reason':'v1 required an immediate axis prefix and omitted host 24 because its group reference followed another host reason. No identity keys, decisions or masks changed.',
            'superseded_projection':str(root/'inputs/identity/uncertain_group_links.csv'),
            'self_tests_before_real_read':checked}
    with (root/'inputs/uncertain_links_receipt_v2.json').open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps({'status':'EXPLICIT_UNCERTAIN_LINKS_V2_PROJECTED','rows':len(links),'groups':result['group_count'],'synthetic_tests':len(checked)}))

if __name__=='__main__':main()
