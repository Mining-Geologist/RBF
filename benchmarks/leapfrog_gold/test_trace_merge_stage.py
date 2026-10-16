"""Native-independent contract checks for compact merge event capture."""
import types
import trace_merge_stage as trace

def test_trace():
    events = []
    class GridSeededDomainer:
        def __init__(self):
            self.domains = {0: [1], 1: [2]}
            self.domainq = [(0, 1)]
            self._next_domain_id = 2
        def merge_domains(self, a, b):
            del self.domains[a], self.domains[b]
            new = self._next_domain_id
            self._next_domain_id += 1
            self.domains[new] = [1, 2]
            return new
    class SubDomainer(GridSeededDomainer):
        pass
    module = types.SimpleNamespace(GridSeededDomainer=GridSeededDomainer,
                                   SubDomainer=SubDomainer)
    methods = trace.attach(module, sink=events.append)
    assert 'GridSeededDomainer.merge_domains' in methods
    assert 'SubDomainer.merge_domains' not in methods
    instance = SubDomainer()
    assert instance.merge_domains(0, 1) == 2
    decisions = [e for e in events if e['type'] == 'return'
                 and e['method'].endswith('merge_domains')]
    assert len(decisions) == 1
    event = decisions[0]
    assert event['created_domains'] == ['2']
    assert event['removed_domains'] == ['0', '1']
    assert event['before']['domains_size'] == 2
    assert event['after']['domains_size'] == 1
    assert trace.attach(module, sink=events.append) == []

def test_exception():
    events = []
    class Example:
        def merge_domains(self):
            raise ValueError("original")
    assert trace.patch_method(Example, "merge_domains", sink=events.append)
    try:
        Example().merge_domains()
    except ValueError as exc:
        assert str(exc) == 'original'
    else:
        raise AssertionError('Original exception suppressed')
    assert events[-1]['type'] == 'exception'

if __name__ == '__main__':
    test_trace()
    test_exception()
    print('PASS merge tracer: unique method hook, domain IDs, queue, exception passthrough')
