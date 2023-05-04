from onset.utilities.logger import logger


def sanitize_magnitude(mag_arg: str) -> int:
    # WARNING. This function has been moved to .src.utilities.tmg.
    # Use that version instead.
    """Converts input magnitude arg into an integer
        Unit identifier is the 4th from list character in the string, mag_arg[-4].
        e.g., 1231904Gbps G is 4th from last.
        this returns 1231904 * 10**9.
    Args:
        mag_arg (str): number joined with either T, G, M, or K.
    Returns:
        int: Value corresponding to the input.
    """

    mag = mag_arg[-4].strip()
    coefficient = int(mag_arg[0:-4])
    logger.debug(coefficient)
    logger.debug(mag)
    exponent = 0
    if mag == 'T':
        exponent = 12
    elif mag == 'G':
        exponent = 9
    elif mag == 'M':
        exponent == 6
    elif mag == 'k':
        exponent == 3
    else:
        raise("ERROR: ill formed magnitude argument. Expected -m <n><T|G|M|k>bps, e.g., 33Gbps")
    result = coefficient * 10 ** exponent
    logger.debug("Result: {}".format(result))
    return result