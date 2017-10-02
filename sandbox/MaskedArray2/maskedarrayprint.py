from numpy import array2string

# in python3 could achieve this with a mixin with new super.

class MaskedFloatFormat(FloatFormat):
    def __call__(self, x, strip_zeros=True):
        if ismasked(x):
            return '--'
        return super(MaskedFloatFormat, self)(x, strip_zeros)

class MaskedIntegerFormat(IntegerFormat):
    def __call__(self, x):
        if ismasked(x):
            return '--'
        return super(MaskedIntegerFormat, self)(x)

class MaskedBoolFormat(BoolFormat):
    def __call__(self, x):
        if ismasked(x):
            return '--'
        return super(MaskedBoolFormat, self)(x)

class MaskedLongFloatFormat(LongFloatFormat):
    def __call__(self, x):
        if ismasked(x):
            return '--'
        return super(MaskedLongFormat, self)(x)

class MaskedLongComplexFormat(LongComplexFormat):
    def __call__(self, x):
        if ismasked(x):
            return '--'
        return super(MaskedLongComplexFormat, self)(x)

class MaskedComplexFormat(ComplexFormat):
    def __call__(self, x):
        if ismasked(x):
            return '--'
        return super(MaskedComplexFormat, self)(x)

class MaskedDatetimeFormat(DatetimeFormat):
    def __call__(self, x):
        if ismasked(x):
            return '--'
        return super(MaskedDatetimeFormat, self)(x)

class MaskedTimedeltaFormat(TimedeltaFormat):
    def __call__(self, x):
        if ismasked(x):
            return '--'
        return super(MaskedDatetimeFormat, self)(x)

_masked_formatter = {
    'bool': MaskedBoolFormat,
    'int': MaskedIntegerFormat,
    'timedelta': MaskedTimedeltaFormat,
    'datetime': MaskedDatetimeFormat,
    'float': MaskedFloatFormat,
    'longfloat': MaskedLongFormat,
    'complexfloat': MaskedComplexFormat,
    'longcomplexfloat': MaskedLongComplexFormat,
    'numpystr': MaskedStrFormat }


_typelessdata = [int_, float_, complex_]
if issubclass(intc, int):
    _typelessdata.append(intc)
if issubclass(longlong, int):
    _typelessdata.append(longlong)

def masked_array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """
    Return the string representation of an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    max_line_width : int, optional
        The maximum number of columns the string should span. Newline
        characters split the string appropriately after array elements.
    precision : int, optional
        Floating point precision. Default is the current printing precision
        (usually 8), which can be altered using `set_printoptions`.
    suppress_small : bool, optional
        Represent very small numbers as zero, default is False. Very small
        is defined by `precision`, if the precision is 8 then
        numbers smaller than 5e-9 are represented as zero.

    Returns
    -------
    string : str
      The string representation of an array.

    See Also
    --------
    array_str, array2string, set_printoptions

    Examples
    --------
    >>> np.array_repr(np.array([1,2]))
    'array([1, 2])'
    >>> np.array_repr(np.ma.array([0.]))
    'MaskedArray([ 0.])'
    >>> np.array_repr(np.array([], np.int32))
    'array([], dtype=int32)'

    >>> x = np.array([1e-6, 4e-7, 2, 3])
    >>> np.array_repr(x, precision=6, suppress_small=True)
    'array([ 0.000001,  0.      ,  2.      ,  3.      ])'

    """
    if type(arr) is not ndarray:
        class_name = type(arr).__name__
    else:
        class_name = "array"

    if arr.size > 0 or arr.shape == (0,):
        lst = array2string(arr, max_line_width, precision, suppress_small,
                           ', ', class_name + "(", formatter=_masked_formatter)
    else:  # show zero-length shape unless it is (0,)
        lst = "[], shape=%s" % (repr(arr.shape),)

    skipdtype = (arr.dtype.type in _typelessdata) and arr.size > 0

    if skipdtype:
        return "%s(%s)" % (class_name, lst)
    else:
        typename = arr.dtype.name
        # Quote typename in the output if it is "complex".
        if typename and not (typename[0].isalpha() and typename.isalnum()):
            typename = "'%s'" % typename

        lf = ' '
        if issubclass(arr.dtype.type, flexible):
            if arr.dtype.names:
                typename = "%s" % str(arr.dtype)
            else:
                typename = "'%s'" % str(arr.dtype)
            lf = '\n'+' '*len(class_name + "(")
        return "%s(%s,%sdtype=%s)" % (class_name, lst, lf, typename)

def masked_array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """
    Return a string representation of the data in an array.

    The data in the array is returned as a single string.  This function is
    similar to `array_repr`, the difference being that `array_repr` also
    returns information on the kind of array and its data type.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.  The
        default is, indirectly, 75.
    precision : int, optional
        Floating point precision.  Default is the current printing precision
        (usually 8), which can be altered using `set_printoptions`.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.

    See Also
    --------
    array2string, array_repr, set_printoptions

    Examples
    --------
    >>> np.array_str(np.arange(3))
    '[0 1 2]'

    """
    return array2string(a, max_line_width, precision, suppress_small, ' ', "",
    formatter=_masked_formatter)

