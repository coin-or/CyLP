#ifndef __PYX_HAVE_API__CyClpSimplex
#define __PYX_HAVE_API__CyClpSimplex
#include "Python.h"

static void (*__pyx_f_12CyClpSimplex_CyPostPrimalRow)(IClpSimplex *) = 0;
#define CyPostPrimalRow __pyx_f_12CyClpSimplex_CyPostPrimalRow
static int (*__pyx_f_12CyClpSimplex_CyPivotIsAcceptable)(IClpSimplex *) = 0;
#define CyPivotIsAcceptable __pyx_f_12CyClpSimplex_CyPivotIsAcceptable

#ifndef __PYX_HAVE_RT_ImportModule
#define __PYX_HAVE_RT_ImportModule
static PyObject *__Pyx_ImportModule(const char *name) {
    PyObject *py_name = 0;
    PyObject *py_module = 0;

    py_name = PyUnicode_FromString(name);
    if (!py_name)
        goto bad;
    py_module = PyImport_Import(py_name);
    Py_DECREF(py_name);
    return py_module;
bad:
    Py_XDECREF(py_name);
    return 0;
}
#endif

#ifndef __PYX_HAVE_RT_ImportFunction
#define __PYX_HAVE_RT_ImportFunction
static int __Pyx_ImportFunction(PyObject *module, const char *funcname, void (**f)(void), const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;
    union {
        void (*fp)(void);
        void *p;
    } tmp;

    d = PyObject_GetAttrString(module, (char *)"__pyx_capi__");
    if (!d)
        goto bad;
    cobj = PyDict_GetItemString(d, funcname);
    if (!cobj) {
        PyErr_Format(PyExc_ImportError,
            "%s does not export expected C function %s",
                PyModule_GetName(module), funcname);
        goto bad;
    }
    if (!PyCapsule_IsValid(cobj, sig)) {
        PyErr_Format(PyExc_TypeError,
            "C function %s.%s has wrong signature (expected %s, got %s)",
             PyModule_GetName(module), funcname, sig, PyCapsule_GetName(cobj));
        goto bad;
    }
    tmp.p = PyCapsule_GetPointer(cobj, sig);
    *f = tmp.fp;
    if (!(*f))
        goto bad;
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(d);
    return -1;
}
#endif

static int import_CyClpSimplex(void) {
  PyObject *module = 0;
  module = __Pyx_ImportModule("CyClpSimplex");
  if (!module) goto bad;
  if (__Pyx_ImportFunction(module, "CyPostPrimalRow", (void (**)(void))&__pyx_f_12CyClpSimplex_CyPostPrimalRow, "void (IClpSimplex *)") < 0) goto bad;
  if (__Pyx_ImportFunction(module, "CyPivotIsAcceptable", (void (**)(void))&__pyx_f_12CyClpSimplex_CyPivotIsAcceptable, "int (IClpSimplex *)") < 0) goto bad;
  Py_DECREF(module); module = 0;
  return 0;
  bad:
  Py_XDECREF(module);
  return -1;
}

#endif /* !__PYX_HAVE_API__CyClpSimplex */
