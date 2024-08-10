def dict_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        retval = {}
        for key in sorted(set(a.keys()) | set(b.keys())):
            if key in a and key in b:
                retval[key] = dict_merge(a[key], b[key])
            elif key in a and key not in b:
                retval[key] = a[key]
            elif key not in a and key in b:
                retval[key] = b[key]
            else:
                assert False, 'Should not reach this line.'  # pragma: no cover
        return retval
    else:
        return b
