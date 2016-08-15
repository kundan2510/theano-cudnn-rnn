#section support_code_struct
cudnnTensorDescriptor_t *APPLY_SPECIFIC(X);
cudnnFilterDescriptor_t *APPLY_SPECIFIC(W);
cudnnTensorDescriptor_t *APPLY_SPECIFIC(output); /* ydesc */
cudnnTensorDescriptor_t *APPLY_SPECIFIC(hxDesc);
cudnnTensorDescriptor_t *APPLY_SPECIFIC(cxDesc);
cudnnTensorDescriptor_t *APPLY_SPECIFIC(hyDesc);
cudnnTensorDescriptor_t *APPLY_SPECIFIC(cyDesc);

#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(X) = NULL;
APPLY_SPECIFIC(W) = NULL;
APPLY_SPECIFIC(output) = NULL;
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(APPLY_SPECIFIC(X))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
	       "(X): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(APPLY_SPECIFIC(output))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(output): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(APPLY_SPECIFIC(hxDesc))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(output): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(APPLY_SPECIFIC(cxDesc))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(output): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(APPLY_SPECIFIC(hyDesc))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(output): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(APPLY_SPECIFIC(cyDesc))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(output): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

if ((APPLY_SPECIFIC(err) = cudnnCreateFilterDescriptor(APPLY_SPECIFIC(W))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate filter descriptor: %s", 
	       cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateDropoutDescriptor(APPLY_SPECIFIC(dropout))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate dropout descriptor: %s", 
	       cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(X) != NULL)
  cudnnDestroyTensorDescriptor(*APPLY_SPECIFIC(X));

if (APPLY_SPECIFIC(output) != NULL)
  cudnnDestroyTensorDescriptor(*APPLY_SPECIFIC(output));

if (APPLY_SPECIFIC(hxDesc) != NULL)
  cudnnDestroyTensorDescriptor(*APPLY_SPECIFIC(hxDesc));

if (APPLY_SPECIFIC(cxDesc) != NULL)
  cudnnDestroyTensorDescriptor(*APPLY_SPECIFIC(cxDesc));

if (APPLY_SPECIFIC(hyDesc) != NULL)
  cudnnDestroyTensorDescriptor(*APPLY_SPECIFIC(hyDesc));

if (APPLY_SPECIFIC(cyDesc) != NULL)
  cudnnDestroyTensorDescriptor(*APPLY_SPECIFIC(cyDesc));

if (APPLY_SPECIFIC(W) != NULL)
  cudnnDestroyFilterDescriptor(*APPLY_SPECIFIC(W));

