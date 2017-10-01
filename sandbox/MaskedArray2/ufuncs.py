###############################################################################
#                                  Ufuncs                                     #
###############################################################################

ufunc_domain = {}
ufunc_fills = {}


class _DomainCheckInterval(object):
    """
    Define a valid interval, so that :

    ``domain_check_interval(a,b)(x) == True`` where
    ``x < a`` or ``x > b``.

    """

    def __init__(self, a, b):
        "domain_check_interval(a,b)(x) = true where x < a or y > b"
        if (a > b):
            (a, b) = (b, a)
        self.a = a
        self.b = b

    def __call__(self, x):
        "Execute the call behavior."
        # nans at masked positions cause RuntimeWarnings, even though
        # they are masked. To avoid this we suppress warnings.
        with np.errstate(invalid='ignore'):
            return umath.logical_or(umath.greater(x, self.b),
                                    umath.less(x, self.a))


class _DomainTan(object):
    """
    Define a valid interval for the `tan` function, so that:

    ``domain_tan(eps) = True`` where ``abs(cos(x)) < eps``

    """

    def __init__(self, eps):
        "domain_tan(eps) = true where abs(cos(x)) < eps)"
        self.eps = eps

    def __call__(self, x):
        "Executes the call behavior."
        with np.errstate(invalid='ignore'):
            return umath.less(umath.absolute(umath.cos(x)), self.eps)


class _DomainSafeDivide(object):
    """
    Define a domain for safe division.

    """

    def __init__(self, tolerance=None):
        self.tolerance = tolerance

    def __call__(self, a, b):
        # Delay the selection of the tolerance to here in order to reduce numpy
        # import times. The calculation of these parameters is a substantial
        # component of numpy's import time.
        if self.tolerance is None:
            self.tolerance = np.finfo(float).tiny
        # don't call ma ufuncs from __array_wrap__ which would fail for scalars
        a, b = np.asarray(a), np.asarray(b)
        with np.errstate(invalid='ignore'):
            return umath.absolute(a) * self.tolerance >= umath.absolute(b)


class _DomainGreater(object):
    """
    DomainGreater(v)(x) is True where x <= v.

    """

    def __init__(self, critical_value):
        "DomainGreater(v)(x) = true where x <= v"
        self.critical_value = critical_value

    def __call__(self, x):
        "Executes the call behavior."
        with np.errstate(invalid='ignore'):
            return umath.less_equal(x, self.critical_value)


class _DomainGreaterEqual(object):
    """
    DomainGreaterEqual(v)(x) is True where x < v.

    """

    def __init__(self, critical_value):
        "DomainGreaterEqual(v)(x) = true where x < v"
        self.critical_value = critical_value

    def __call__(self, x):
        "Executes the call behavior."
        with np.errstate(invalid='ignore'):
            return umath.less(x, self.critical_value)


class _MaskedUFunc(object):
    def __init__(self, ufunc):
        self.f = ufunc
        self.__doc__ = ufunc.__doc__
        self.__name__ = ufunc.__name__

    def __str__(self):
        return "Masked version of {}".format(self.f)


class _MaskedUnaryOperation(_MaskedUFunc):
    """
    Defines masked version of unary operations, where invalid values are
    pre-masked.

    Parameters
    ----------
    mufunc : callable
        The function for which to define a masked version. Made available
        as ``_MaskedUnaryOperation.f``.
    fill : scalar, optional
        Filling value, default is 0.
    domain : class instance
        Domain for the function. Should be one of the ``_Domain*``
        classes. Default is None.

    """

    def __init__(self, mufunc, fill=0, domain=None):
        super(_MaskedUnaryOperation, self).__init__(mufunc)
        self.fill = fill
        self.domain = domain
        ufunc_domain[mufunc] = domain
        ufunc_fills[mufunc] = fill

    def __call__(self, a, *args, **kwargs):
        """
        Execute the call behavior.

        """
        d = getdata(a)
        # Deal with domain
        if self.domain is not None:
            # Case 1.1. : Domained function
            # nans at masked positions cause RuntimeWarnings, even though
            # they are masked. To avoid this we suppress warnings.
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.f(d, *args, **kwargs)
            # Make a mask
            m = ~umath.isfinite(result)
            m |= self.domain(d)
            m |= getmask(a)
        else:
            # Case 1.2. : Function without a domain
            # Get the result and the mask
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.f(d, *args, **kwargs)
            m = getmask(a)

        if not result.ndim:
            # Case 2.1. : The result is scalarscalar
            if m:
                return masked
            return result

        if m is not nomask:
            # Case 2.2. The result is an array
            # We need to fill the invalid data back w/ the input Now,
            # that's plain silly: in C, we would just skip the element and
            # keep the original, but we do have to do it that way in Python

            # In case result has a lower dtype than the inputs (as in
            # equal)
            try:
                np.copyto(result, d, where=m)
            except TypeError:
                pass
        # Transform to
        masked_result = result.view(get_masked_subclass(a))
        masked_result._mask = m
        masked_result._update_from(a)
        return masked_result


class _MaskedBinaryOperation(_MaskedUFunc):
    """
    Define masked version of binary operations, where invalid
    values are pre-masked.

    Parameters
    ----------
    mbfunc : function
        The function for which to define a masked version. Made available
        as ``_MaskedBinaryOperation.f``.
    domain : class instance
        Default domain for the function. Should be one of the ``_Domain*``
        classes. Default is None.
    fillx : scalar, optional
        Filling value for the first argument, default is 0.
    filly : scalar, optional
        Filling value for the second argument, default is 0.

    """

    def __init__(self, mbfunc, fillx=0, filly=0):
        """
        abfunc(fillx, filly) must be defined.

        abfunc(x, filly) = x for all x to enable reduce.

        """
        super(_MaskedBinaryOperation, self).__init__(mbfunc)
        self.fillx = fillx
        self.filly = filly
        ufunc_domain[mbfunc] = None
        ufunc_fills[mbfunc] = (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        """
        Execute the call behavior.

        """
        # Get the data, as ndarray
        (da, db) = (getdata(a), getdata(b))
        # Get the result
        with np.errstate():
            np.seterr(divide='ignore', invalid='ignore')
            result = self.f(da, db, *args, **kwargs)
        # Get the mask for the result
        (ma, mb) = (getmask(a), getmask(b))
        if ma is nomask:
            if mb is nomask:
                m = nomask
            else:
                m = umath.logical_or(getmaskarray(a), mb)
        elif mb is nomask:
            m = umath.logical_or(ma, getmaskarray(b))
        else:
            m = umath.logical_or(ma, mb)

        # Case 1. : scalar
        if not result.ndim:
            if m:
                return masked
            return result

        # Case 2. : array
        # Revert result to da where masked
        if m is not nomask and m.any():
            # any errors, just abort; impossible to guarantee masked values
            try:
                np.copyto(result, da, casting='unsafe', where=m)
            except Exception:
                pass

        # Transforms to a (subclass of) MaskedArray
        masked_result = result.view(get_masked_subclass(a, b))
        masked_result._mask = m
        if isinstance(a, MaskedArray):
            masked_result._update_from(a)
        elif isinstance(b, MaskedArray):
            masked_result._update_from(b)
        return masked_result

    def reduce(self, target, axis=0, dtype=None):
        """
        Reduce `target` along the given `axis`.

        """
        tclass = get_masked_subclass(target)
        m = getmask(target)
        t = filled(target, self.filly)
        if t.shape == ():
            t = t.reshape(1)
            if m is not nomask:
                m = make_mask(m, copy=1)
                m.shape = (1,)

        if m is nomask:
            tr = self.f.reduce(t, axis)
            mr = nomask
        else:
            tr = self.f.reduce(t, axis, dtype=dtype or t.dtype)
            mr = umath.logical_and.reduce(m, axis)

        if not tr.shape:
            if mr:
                return masked
            else:
                return tr
        masked_tr = tr.view(tclass)
        masked_tr._mask = mr
        return masked_tr

    def outer(self, a, b):
        """
        Return the function applied to the outer product of a and b.

        """
        (da, db) = (getdata(a), getdata(b))
        d = self.f.outer(da, db)
        ma = getmask(a)
        mb = getmask(b)
        if ma is nomask and mb is nomask:
            m = nomask
        else:
            ma = getmaskarray(a)
            mb = getmaskarray(b)
            m = umath.logical_or.outer(ma, mb)
        if (not m.ndim) and m:
            return masked
        if m is not nomask:
            np.copyto(d, da, where=m)
        if not d.shape:
            return d
        masked_d = d.view(get_masked_subclass(a, b))
        masked_d._mask = m
        return masked_d

    def accumulate(self, target, axis=0):
        """Accumulate `target` along `axis` after filling with y fill
        value.

        """
        tclass = get_masked_subclass(target)
        t = filled(target, self.filly)
        result = self.f.accumulate(t, axis)
        masked_result = result.view(tclass)
        return masked_result



class _DomainedBinaryOperation(_MaskedUFunc):
    """
    Define binary operations that have a domain, like divide.

    They have no reduce, outer or accumulate.

    Parameters
    ----------
    mbfunc : function
        The function for which to define a masked version. Made available
        as ``_DomainedBinaryOperation.f``.
    domain : class instance
        Default domain for the function. Should be one of the ``_Domain*``
        classes.
    fillx : scalar, optional
        Filling value for the first argument, default is 0.
    filly : scalar, optional
        Filling value for the second argument, default is 0.

    """

    def __init__(self, dbfunc, domain, fillx=0, filly=0):
        """abfunc(fillx, filly) must be defined.
           abfunc(x, filly) = x for all x to enable reduce.
        """
        super(_DomainedBinaryOperation, self).__init__(dbfunc)
        self.domain = domain
        self.fillx = fillx
        self.filly = filly
        ufunc_domain[dbfunc] = domain
        ufunc_fills[dbfunc] = (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        "Execute the call behavior."
        # Get the data
        (da, db) = (getdata(a), getdata(b))
        # Get the result
        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(da, db, *args, **kwargs)
        # Get the mask as a combination of the source masks and invalid
        m = ~umath.isfinite(result)
        m |= getmask(a)
        m |= getmask(b)
        # Apply the domain
        domain = ufunc_domain.get(self.f, None)
        if domain is not None:
            m |= domain(da, db)
        # Take care of the scalar case first
        if (not m.ndim):
            if m:
                return masked
            else:
                return result
        # When the mask is True, put back da if possible
        # any errors, just abort; impossible to guarantee masked values
        try:
            np.copyto(result, 0, casting='unsafe', where=m)
            # avoid using "*" since this may be overlaid
            masked_da = umath.multiply(m, da)
            # only add back if it can be cast safely
            if np.can_cast(masked_da.dtype, result.dtype, casting='safe'):
                result += masked_da
        except Exception:
            pass

        # Transforms to a (subclass of) MaskedArray
        masked_result = result.view(get_masked_subclass(a, b))
        masked_result._mask = m
        if isinstance(a, MaskedArray):
            masked_result._update_from(a)
        elif isinstance(b, MaskedArray):
            masked_result._update_from(b)
        return masked_result


# Unary ufuncs
exp = _MaskedUnaryOperation(umath.exp)
conjugate = _MaskedUnaryOperation(umath.conjugate)
sin = _MaskedUnaryOperation(umath.sin)
cos = _MaskedUnaryOperation(umath.cos)
tan = _MaskedUnaryOperation(umath.tan)
arctan = _MaskedUnaryOperation(umath.arctan)
arcsinh = _MaskedUnaryOperation(umath.arcsinh)
sinh = _MaskedUnaryOperation(umath.sinh)
cosh = _MaskedUnaryOperation(umath.cosh)
tanh = _MaskedUnaryOperation(umath.tanh)
abs = absolute = _MaskedUnaryOperation(umath.absolute)
angle = _MaskedUnaryOperation(angle)  # from numpy.lib.function_base
fabs = _MaskedUnaryOperation(umath.fabs)
negative = _MaskedUnaryOperation(umath.negative)
floor = _MaskedUnaryOperation(umath.floor)
ceil = _MaskedUnaryOperation(umath.ceil)
around = _MaskedUnaryOperation(np.round_)
logical_not = _MaskedUnaryOperation(umath.logical_not)

# Domained unary ufuncs
sqrt = _MaskedUnaryOperation(umath.sqrt, 0.0,
                             _DomainGreaterEqual(0.0))
log = _MaskedUnaryOperation(umath.log, 1.0,
                            _DomainGreater(0.0))
log2 = _MaskedUnaryOperation(umath.log2, 1.0,
                             _DomainGreater(0.0))
log10 = _MaskedUnaryOperation(umath.log10, 1.0,
                              _DomainGreater(0.0))
tan = _MaskedUnaryOperation(umath.tan, 0.0,
                            _DomainTan(1e-35))
arcsin = _MaskedUnaryOperation(umath.arcsin, 0.0,
                               _DomainCheckInterval(-1.0, 1.0))
arccos = _MaskedUnaryOperation(umath.arccos, 0.0,
                               _DomainCheckInterval(-1.0, 1.0))
arccosh = _MaskedUnaryOperation(umath.arccosh, 1.0,
                                _DomainGreaterEqual(1.0))
arctanh = _MaskedUnaryOperation(umath.arctanh, 0.0,
                                _DomainCheckInterval(-1.0 + 1e-15, 1.0 - 1e-15))

# Binary ufuncs
add = _MaskedBinaryOperation(umath.add)
subtract = _MaskedBinaryOperation(umath.subtract)
multiply = _MaskedBinaryOperation(umath.multiply, 1, 1)
arctan2 = _MaskedBinaryOperation(umath.arctan2, 0.0, 1.0)
equal = _MaskedBinaryOperation(umath.equal)
equal.reduce = None
not_equal = _MaskedBinaryOperation(umath.not_equal)
not_equal.reduce = None
less_equal = _MaskedBinaryOperation(umath.less_equal)
less_equal.reduce = None
greater_equal = _MaskedBinaryOperation(umath.greater_equal)
greater_equal.reduce = None
less = _MaskedBinaryOperation(umath.less)
less.reduce = None
greater = _MaskedBinaryOperation(umath.greater)
greater.reduce = None
logical_and = _MaskedBinaryOperation(umath.logical_and)
alltrue = _MaskedBinaryOperation(umath.logical_and, 1, 1).reduce
logical_or = _MaskedBinaryOperation(umath.logical_or)
sometrue = logical_or.reduce
logical_xor = _MaskedBinaryOperation(umath.logical_xor)
bitwise_and = _MaskedBinaryOperation(umath.bitwise_and)
bitwise_or = _MaskedBinaryOperation(umath.bitwise_or)
bitwise_xor = _MaskedBinaryOperation(umath.bitwise_xor)
hypot = _MaskedBinaryOperation(umath.hypot)

# Domained binary ufuncs
divide = _DomainedBinaryOperation(umath.divide, _DomainSafeDivide(), 0, 1)
true_divide = _DomainedBinaryOperation(umath.true_divide,
                                       _DomainSafeDivide(), 0, 1)
floor_divide = _DomainedBinaryOperation(umath.floor_divide,
                                        _DomainSafeDivide(), 0, 1)
remainder = _DomainedBinaryOperation(umath.remainder,
                                     _DomainSafeDivide(), 0, 1)
fmod = _DomainedBinaryOperation(umath.fmod, _DomainSafeDivide(), 0, 1)
mod = _DomainedBinaryOperation(umath.mod, _DomainSafeDivide(), 0, 1)

