#ifndef UTILS_H
#define UTILS_H

#include "common.h"

#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>

template<typename iT, typename vT>
double getB(const iT m, const iT nnz){
    return (double)((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
}

template<typename iT>
double getFLOP(const iT nnz){
    return (double)(2 * nnz);
}

#endif // UTILS_H
