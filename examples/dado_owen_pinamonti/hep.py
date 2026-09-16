import json
def rows(fn):
    j=json.load(open(fn)); hdr=[h['name'] for h in j['headers']]
    return hdr, [[c.get('value') for c in r['x']] + [c.get('value') for c in r['y']] for r in j['values']]
