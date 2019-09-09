def div2(x):
    return int(x/2)


def dfloor(x, d):
    return int(x - (x % d))


def padint(val, places=4):
    assert(val >= 0)
    vlen = len(str(val))
    places = max(vlen, places)
    diff = places - vlen
    if diff == 1:
        return '0' + str(val)
    elif diff == 2:
        return '00' + str(val)
    elif diff == 3:
        return '000' + str(val)
    elif diff == 4:
        return '0000' + str(val)
    elif diff == 5:
        return '00000' + str(val)
    else:
        return str(val)
