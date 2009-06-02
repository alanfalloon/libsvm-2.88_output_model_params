"""
This provides a lower-level interface to the svm module.

This SVM solver accepts precomputed kernel matrices (gram matrix)
instead of kernel functions which are a more traditional
interface. Additionally, it returns the underlying alpha and beta
parameters of the result instead of a model that can be queried.

This wrapper provides a function svm that conforms to that more basic
interface.

It also provides a function to get the raw parameters from an
svm.model instance that used precomputed kernels
"""
from numpy import zeros, arange, hstack, asarray, double, int
from svm import svm_problem, svm_parameter, svm_model, PRECOMPUTED
import svmc

def svm(y,K,**param_kw):
    """
    Solve the SVM problem. Return ``(alpha, b)``

    `y`
      labels
    `K`
      precopmuted kernel matrix

    Additional keyword arguments are passed on as svm parameters to
    the model.

    The wrapper is needed to precondition the precomputed matrix for
    use with libsvm, and to extract the model parameters and convert
    them into the canonical weight vector plus scalar offset. Normally
    libsvm hides these model paramters, preferring instead to provide
    a high-level model object that can be queried for results.

    """
    i = arange(1,len(K)+1).reshape((-1,1))
    X = hstack((i, K))
    y = asarray(y,dtype=double)
    X = asarray(X,dtype=double)
    prob = svm_problem(y,X)
    param = svm_parameter(kernel_type=PRECOMPUTED,**param_kw)
    model = svm_model(prob, param)
    return get_alpha_b(model)

def get_alpha_b(model):
    """
    Get the ``(alpha, b)`` parameters from a model
    """
    y = model.prob.y_array
    len_y = model.prob.size
    y_0 = svmc.double_getitem(y,0)
    coefs, perm, rho = get_model_params(model)
    minp = min(perm)
    alpha = zeros(len_y)
    alpha[perm-minp] = coefs
    return (alpha, -rho)

def get_model_params(model):
    """
    Extract the alpha and b parameters from the SVM model.

    returns (alpha, b)
    """
    rho = svmc.svm_get_model_rho(model.model)
    n = svmc.svm_get_model_num_coefs(model.model)
    coefs_dblarr = svmc.new_double(n)
    perm_intarr = svmc.new_int(n)
    try:
        svmc.svm_get_model_coefs(model.model,coefs_dblarr)
        svmc.svm_get_model_perm(model.model,perm_intarr)
        coefs = zeros(n,dtype=double)
        perm = zeros(n,dtype=int)
        for i in range(n):
            coefs[i] = svmc.double_getitem(coefs_dblarr,i)
            perm[i] = svmc.int_getitem(perm_intarr,i)
    finally:
        svmc.delete_double(coefs_dblarr)
        svmc.delete_int(perm_intarr)
    return (coefs, perm, rho)
