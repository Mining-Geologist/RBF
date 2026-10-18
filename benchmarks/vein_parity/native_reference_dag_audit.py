"""Static native Leapfrog project dependency audit, no vendor module imports.

Usage: python native_reference_dag_audit.py /path/to/Vein_Benchmark.aproj
Reads owner's SQLite/pickle persistent-object metadata with pickletools only.
Never executes vendor classes or calls pickle.loads. Input files are not bundled.
"""
import argparse
import json
import pickletools
import sqlite3
from pathlib import Path


def streams(blob):
    first = list(pickletools.genops(blob))
    cut = first[-1][2] + 1
    return first, list(pickletools.genops(blob[cut:]))


class ProjectGraph:
    def __init__(self, project):
        uri = Path(project).resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as db:
            self.blobs = dict(db.execute('SELECT oid, data FROM data'))
        self.classes, self.ops = {}, {}
        for oid, blob in self.blobs.items():
            try:
                cls, body = streams(blob)
                if len(cls) > 4 and cls[2][0].name.endswith('UNICODE'):
                    self.classes[oid] = str(cls[2][1]) + '.' + str(cls[4][1])
                    self.ops[oid] = body
            except Exception:
                pass

    def field_oid(self, oid, field):
        operations = self.ops.get(oid, ())
        for i, (op, value, _) in enumerate(operations):
            if op.name.endswith('UNICODE') and value == field:
                for nextop, nextvalue, _ in operations[i+1:i+7]:
                    if nextop.name.endswith('BINBYTES') and isinstance(nextvalue, bytes) and len(nextvalue) == 8:
                        return int.from_bytes(nextvalue, 'big')
                    if nextop.name == 'NONE':
                        return None
        return None

    def flag(self, oid, field):
        operations = self.ops.get(oid, ())
        for i, (op, value, _) in enumerate(operations):
            if op.name.endswith('UNICODE') and value == field:
                for nextop, _, _ in operations[i+1:i+6]:
                    if nextop.name in ('NEWTRUE', 'NEWFALSE'):
                        return nextop.name == 'NEWTRUE'
        return None

    def inputs(self, oid):
        mapping_id = self.field_oid(oid, '_inputs')
        if mapping_id is None:
            return []
        operations = self.ops.get(mapping_id, ())
        found = []
        for i, (op, _, _) in enumerate(operations):
            if op.name != 'BINPERSID':
                continue
            for before, value, _ in reversed(operations[max(0, i-18):i]):
                if before.name in ('BINBYTES', 'SHORT_BINBYTES') and isinstance(value, bytes) and len(value) == 8:
                    child = int.from_bytes(value, 'big')
                    if child in self.classes:
                        found.append(child)
                    break
        return sorted(set(found))

    def find(self, suffix):
        return sorted(k for k, cls in self.classes.items() if cls.endswith(suffix))


def audit(project):
    g = ProjectGraph(project)
    references = g.find('.VeinReferenceSurfaceBlock')
    generators = g.find('.VeinWallPointsGeneratorBlock')
    models = []
    for oid in references:
        users = [gen for gen in generators if oid in g.inputs(gen)]
        models.append({
            'reference_oid': oid,
            'input_classes': sorted(g.classes[d].split('.')[-1] for d in g.inputs(oid)),
            'midpoint_input_oid': g.field_oid(oid, 'points'),
            'hw_points': g.field_oid(oid, 'hw_points'),
            'fw_points': g.field_oid(oid, 'fw_points'),
            'wall_generators': [
                {'oid': gen, 'pinchout': g.flag(gen, 'is_pinch_out')} for gen in sorted(users)
            ],
        })
    return {'reference_count': len(references),
            'wall_generator_count': len(generators), 'models': models}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project')
    print(json.dumps(audit(parser.parse_args().project), indent=2))
