"""Project explicit frozen uncertain group references; do not infer group hashes."""
import csv,hashlib,io,json,re
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    receipt=json.loads((root/'inputs/projection_receipt.json').read_text('utf-8'))
    source=receipt['sources']['parent_appearance_keys']
    payload=Path(source['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=source['sha256']:
        raise RuntimeError('frozen source drift before necessary identity-only linkage projection')
    pattern=re.compile(r'(?:^| \| )(host|guest):GROUP_REVIEW_UNCERTAIN:(v2_parent_neutralized:(host|guest):[0-9a-f]{16})(?= \| |$)')
    links=set()
    for row in csv.DictReader(io.StringIO(payload.decode('utf-8-sig'),newline='')):
        if row['layer']!='v2_reviewed_only_parent':
            continue
        for axis,group,group_axis in pattern.findall(row['uncertain_reason']):
            if axis!=group_axis:
                raise RuntimeError('frozen reason contains inconsistent group axis')
            links.add((axis,row[axis+'_id'],group))
    output=root/'inputs/identity/uncertain_group_links.csv'
    with output.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(['role','raw_id','review_group_id'])
        writer.writerows(sorted(links,key=lambda r:(r[0],int(r[1]),r[2])))
    result={'source':source,'projection':str(output),'rows':len(links),
            'group_count':len({row[2] for row in links}),
            'method':'Literal GROUP_REVIEW_UNCERTAIN references from frozen primary identity reasons; no inferred hash or chemical judgment.'}
    with (root/'inputs/uncertain_links_receipt.json').open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps({'status':'EXPLICIT_UNCERTAIN_LINKS_PROJECTED','rows':len(links),'groups':result['group_count']}))


if __name__=='__main__':
    main()
