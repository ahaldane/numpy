#ifndef _NPY_ARRAYMAPPING_H_
#define _NPY_ARRAYMAPPING_H_

extern NPY_NO_EXPORT PyMappingMethods array_as_mapping;


/*
 * Struct into which indices are parsed.
 * I.e. integer ones should only be parsed once, slices and arrays
 * need to be validated later and for the ellipsis we need to find how
 * many slices it represents.
 */
typedef struct {
    /*
     * Object of index: slice, array, or NULL. Owns a reference.
     */
    PyObject *object;
    /*
     * For HAS_INTEGER, int_index is the value of the integer index. For
     * HAS_ELLIPSIS, ellipsis_slices is the number of slices the ellipsis is
     * worth. For HAS_FANCY, fancy_bool is -1 for an integer array, or the
     * length of the array for boolean arrays.
     */
    union{
        npy_intp int_index;
        int ellipsis_slices;
        npy_intp fancy_bool;
    };

    /* kind of index, see constants in mapping.c */
    int type;
} npy_index_info;


NPY_NO_EXPORT Py_ssize_t
array_length(PyArrayObject *self);

NPY_NO_EXPORT PyObject *
array_item_asarray(PyArrayObject *self, npy_intp i);

NPY_NO_EXPORT PyObject *
array_item_asscalar(PyArrayObject *self, npy_intp i);

NPY_NO_EXPORT PyObject *
array_item(PyArrayObject *self, Py_ssize_t i);

NPY_NO_EXPORT PyObject *
array_subscript_asarray(PyArrayObject *self, PyObject *op);

NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op);

NPY_NO_EXPORT int
array_assign_item(PyArrayObject *self, Py_ssize_t i, PyObject *v);

/*
 * Prototypes for Mapping calls --- not part of the C-API
 * because only useful as part of a getitem call.
 */
NPY_NO_EXPORT void
PyArray_MapIterReset(PyArrayMapIterObject *mit);

NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *mit);

NPY_NO_EXPORT int
PyArray_MapIterCheckIndices(PyArrayMapIterObject *mit);

NPY_NO_EXPORT void
PyArray_MapIterSwapAxes(PyArrayMapIterObject *mit, PyArrayObject **ret, int getmap);

NPY_NO_EXPORT PyObject*
PyArray_MapIterNew(npy_index_info *indices , int index_num, int index_type,
                   int ndim, int fancy_ndim,
                   PyArrayObject *arr, PyArrayObject *subspace,
                   npy_uint32 subspace_iter_flags, npy_uint32 subspace_flags,
                   npy_uint32 extra_op_flags, PyArrayObject *extra_op,
                   PyArray_Descr *extra_op_dtype);
#endif
