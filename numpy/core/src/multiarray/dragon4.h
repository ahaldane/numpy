/*
 * Copyright (c) 2014 Ryan Juckett
 * http://www.ryanjuckett.com/
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source
 *    distribution.
 */

/*
 * This file contains a modified version of Ryan Juckett's Dragon4
 * implementation, which has been ported from C++ to C and which has
 * modifications specific to printing floats in numpy.
 */

#ifndef _NPY_DRAGON4_H_
#define _NPY_DRAGON4_H_

#include "Python.h"
#include "structmember.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "numpy/arrayscalars.h"
#include "numpy/npy_common.h"


#define NPY_HALF_BINFMT_NAME IEEE_binary16

/* NPY_FLOAT_BINFMT_NAME and NPY_DOUBLE_BINFMT defined in npy_common.h */

#if defined(HAVE_LDOUBLE_IEEE_QUAD_BE)
    #define NPY_LONGDOUBLE_BINFMT_NAME IEEE_binary128_be
#elif defined(HAVE_LDOUBLE_IEEE_QUAD_LE)
    #define NPY_LONGDOUBLE_BINFMT_NAME IEEE_binary128_le
#elif (defined(HAVE_LDOUBLE_IEEE_DOUBLE_LE) || \
       defined(HAVE_LDOUBLE_IEEE_DOUBLE_BE))
    #define NPY_LONGDOUBLE_BINFMT_NAME IEEE_binary64
#elif defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE)
    #define NPY_LONGDOUBLE_BINFMT_NAME IBM_double_double_le
#elif defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE)
    #define NPY_LONGDOUBLE_BINFMT_NAME IBM_double_double_be
#elif defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE)
    #define NPY_LONGDOUBLE_BINFMT_NAME Intel_extended96
#elif defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE)
    #define NPY_LONGDOUBLE_BINFMT_NAME Intel_extended128
#elif defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE)
    #define NPY_LONGDOUBLE_BINFMT_NAME Motorola_extended96
#else
    #error No long double representation defined
#endif

typedef enum DigitMode
{
    /* Round digits to print shortest uniquely identifiable number. */
    DigitMode_Unique,
    /* Output the digits of the number as if with infinite precision */
    DigitMode_Exact,
} DigitMode;

typedef enum CutoffMode
{
    /* up to cutoffNumber significant digits */
    CutoffMode_TotalLength,
    /* up to cutoffNumber significant digits past the decimal point */
    CutoffMode_FractionLength,
} CutoffMode;

typedef enum TrimMode
{
    TrimMode_None,         /* don't trim zeros, always leave a decimal point */
    TrimMode_LeaveOneZero, /* trim all but the zero before the decimal point */
    TrimMode_Zeros,        /* trim all trailing zeros, leave decimal point */
    TrimMode_DptZeros,     /* trim trailing zeros & trailing decimal point */
} TrimMode;

#define make_dragon4_typedecl(Type, npy_type, format) \
\
PyObject *\
Dragon4_Positional_##Type(npy_type *val, DigitMode digit_mode,\
                   CutoffMode cutoff_mode, int precision,\
                   int sign, TrimMode trim, int pad_left, int pad_right);\
PyObject *\
Dragon4_Scientific_##Type(npy_type *val, DigitMode digit_mode, int precision,\
                   int sign, TrimMode trim, int pad_left, int exp_digits);

make_dragon4_typedecl(Half, npy_half, NPY_HALF_BINFMT_NAME)
make_dragon4_typedecl(Float, npy_float, NPY_FLOAT_BINFMT_NAME)
make_dragon4_typedecl(Double, npy_double, NPY_DOUBLE_BINFMT_NAME)
make_dragon4_typedecl(LongDouble, npy_longdouble, NPY_LONGDOUBLE_BINFMT_NAME)

#undef make_dragon4_typedecl

PyObject *
Dragon4_Positional(PyObject *obj, DigitMode digit_mode, CutoffMode cutoff_mode,
                   int precision, int sign, TrimMode trim, int pad_left,
                   int pad_right);

PyObject *
Dragon4_Scientific(PyObject *obj, DigitMode digit_mode, int precision,
                   int sign, TrimMode trim, int pad_left, int exp_digits);

#endif

