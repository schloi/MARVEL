
import colorramp

def normalize(val, min, max):
    assert(min <= max)

    if val < min:
        val = min
    elif val > max:
        val = max

    if ( max - min ) != 0:
        return float( val - min ) / ( max - min )

    return 1

def map(val, min, max):
    val = int( normalize(val, min, max) * ( len(colorramp.RAMP) - 1 ) )
    return colorramp.RAMP[val]
