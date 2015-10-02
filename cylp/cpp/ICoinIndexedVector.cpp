#include "ICoinIndexedVector.hpp"

// define PyInt_* macros for Python 3.x
#ifndef PyInt_Check
#define PyInt_Check             PyLong_Check
#define PyInt_FromLong          PyLong_FromLong
#define PyInt_AsLong            PyLong_AsLong
#define PyInt_Type              PyLong_Type
#endif

ICoinIndexedVector::ICoinIndexedVector(){
    _import_array();
}


PyObject* ICoinIndexedVector::getIndicesNPArray(){

    npy_intp dims = this->getNumElements();
    PyObject *Arr = PyArray_SimpleNewFromData(1, &dims, 
            PyArray_INT32, 
            this->getIndices());
    return Arr;
}

PyObject* ICoinIndexedVector::getDenseVectorNPArray(){

    npy_intp dims = this->capacity();
    PyObject *Arr = PyArray_SimpleNewFromData(1, &dims, 
            PyArray_DOUBLE, 
            this->denseVector());
    return Arr;
}

double ICoinIndexedVector::getItem(int n){
    return operator[](n);
}

void ICoinIndexedVector::setItem(int index, double element){
    //operator[](n) = value;
    #ifndef COIN_FAST_CODE
      if ( index < 0 ) 
        throw CoinError("index < 0" , "setElement", "CoinIndexedVector");
    #endif
      if (index >= capacity())
        reserve(index+1);
      if (denseVector()[index]) {
        //element += denseVector()[index];
        if (fabs(element)>= COIN_INDEXED_TINY_ELEMENT) {
          denseVector()[index] = element;
        } else {
          denseVector()[index] = COIN_INDEXED_REALLY_TINY_ELEMENT;
        }
      } else if (fabs(element)>= COIN_INDEXED_TINY_ELEMENT) {
        getIndices()[getNumElements()] = index;
        setNumElements(getNumElements() + 1);
        assert (getNumElements()<=capacity());
        denseVector()[index] = element;
       }
    
}

void ICoinIndexedVector::assign(PyObject* ind, PyObject* other){
    int other_is_num = 0, other_is_list = 0, other_is_array = 0;
    int ind_is_num = 0, ind_is_list = 0, ind_is_array = 0;
    double val = 0.0;
    int idx = 0;
    int ind_i;

    //_import_array();

    if( PyInt_Check(other) ) {
        val = (double)PyInt_AsLong(other);
        other_is_num = 1;
    } else if( PyFloat_Check(other) ) {
        val = PyFloat_AsDouble(other);
        other_is_num = 1;
    } else if (PyList_Check(other))
        other_is_list = 1;
    else if (PyArray_Check(other))
        other_is_array = 1;
    else{
        PyErr_SetString(PyExc_ValueError, "Unknown type for rhs.");
        return;
    }


    if( PyInt_Check(ind) ) {
        idx = (double)PyInt_AsLong(ind);
        ind_is_num = 1;
    } else if( PyFloat_Check(ind) ) {
        idx = PyFloat_AsDouble(ind);
        ind_is_num = 1;
    } else if (PyList_Check(ind))
        ind_is_list = 1;
    else if (PyArray_Check(ind))
        ind_is_array = 1;
    else{
        PyErr_SetString(PyExc_ValueError, "Unknown type for index set");
        return;
    }

    if (ind_is_num && other_is_num){
        setItem(idx, val);
        return;
    }

    // index set is a list, 3 possibilities for the rhs: number, list, array
    else if (ind_is_list){
        Py_ssize_t ind_length = PyList_Size(ind);
        if (other_is_num){
            for (int i = 0 ; i < ind_length; i++){
                ind_i = PyArray_PyIntAsInt(PyList_GetItem(ind, (Py_ssize_t)i));
                setItem(ind_i, PyFloat_AsDouble(other));
            }
        }
        else if (other_is_list){
            Py_ssize_t other_length = PyList_Size(other);
            if (ind_length != other_length){
                PyErr_SetString(PyExc_ValueError, 
                        "Index set and right hand side values must have the same size");
                return;
            }
            for (int i = 0 ; i < ind_length ; i++){
                ind_i = PyArray_PyIntAsInt(PyList_GetItem(ind, (Py_ssize_t)i));
                setItem(ind_i, 
                        PyFloat_AsDouble(PyList_GetItem(other, (Py_ssize_t)i)));
            }
        }else if (other_is_array){
            npy_intp other_length = PyArray_DIM(other, 0);
            PyObject* iterator = PyArray_IterNew(other);
            if (ind_length != other_length){
                PyErr_SetString(PyExc_ValueError, 
                        "Index set and right hand side values must have the same size");
                return;
            }
            //int counter = 0;
            for (int i = 0 ; i < other_length ; i++){
                //while( PyArray_ITER_NOTDONE(iterator) ) {
                val = *(double*)PyArray_ITER_DATA(iterator);
                ind_i = PyArray_PyIntAsInt(PyList_GetItem(ind, (Py_ssize_t)i));
                setItem(ind_i, val);
                PyArray_ITER_NEXT(iterator);
                //counter++;
            }
            }

        } 


        // If the index set is a numpy array 3 possibilities for the rhs: number, list, array 
        else if (ind_is_array){
            npy_intp ind_length = PyArray_DIM(ind, 0);
            PyObject* ind_iterator = PyArray_IterNew(ind);

            if (other_is_num){
                while( PyArray_ITER_NOTDONE(ind_iterator) ){
                    //for (int i = 0 ; i < ind_length; i++){
                    ind_i = *(int*)PyArray_ITER_DATA(ind_iterator);
                    setItem(ind_i, PyFloat_AsDouble(other));
                    PyArray_ITER_NEXT(ind_iterator);
                }
                }
                else if (other_is_list){
                    Py_ssize_t other_length = PyList_Size(other);
                    if (ind_length != other_length){
                        PyErr_SetString(PyExc_ValueError, 
                                "Index set and right hand side values must have the same size");
                        return;
                    }
                    for (int i = 0 ; i < ind_length ; i++){
                        ind_i = *(int*)PyArray_ITER_DATA(ind_iterator);
                        setItem(ind_i, 
                                PyFloat_AsDouble(PyList_GetItem(other, (Py_ssize_t)i)));
                        PyArray_ITER_NEXT(ind_iterator);
                    }
                }else if (other_is_array){
                    npy_intp other_length = PyArray_DIM(other, 0);
                    PyObject* iterator = PyArray_IterNew(other);
                    if (ind_length != other_length){
                        PyErr_SetString(PyExc_ValueError, 
                                "Index set and right hand side values must have the same size");
                        return;
                    }
                    //int counter = 0;
                    for (int i = 0 ; i < other_length ; i++){
                        //while( PyArray_ITER_NOTDONE(iterator) ) {
                        val = *(double*)PyArray_ITER_DATA(iterator);
                        ind_i = *(int*)PyArray_ITER_DATA(ind_iterator);
                        setItem(ind_i, val);
                        PyArray_ITER_NEXT(iterator);
                        PyArray_ITER_NEXT(ind_iterator);
                        // counter++;
                    }
                    }

                } 

            }
