#section support_code_struct
cudnnDropoutDescriptor_t *APPLY_SPECIFIC(dropout);

#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(err) = cudnnCreateDropoutDescriptor(dropDesc);

if (APPLY_SPECIFIC(err) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not create dropout descriptor "
               "descriptor: %s", cudnnGetErrorString(err));
  FAIL;
}

#section support_code_apply

int APPLY_SPECIFIC(rnn_desc)(cudnnRNNDescriptor_t *desc) {
  int hiddenSize = HIDDEN_SIZE;
  int numLayers = NUM_LAYERS;
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  cudnnDirectionMode_t direction = DIRECTION;
  cudnnRNNMode_t mode = MODE;
  cudnnDataType_t dataType = PRECISION;

  APPLY_SPECIFIC(err) = cudnnCreateRNNDescriptor(desc);
  if (APPLY_SPECIFIC(err) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not create RNN descriptor"
                 "descriptor: %s", cudnnGetErrorString(err));
    return -1;
  }

  cudnnRNNDescriptor_t * dropDesc = APPLY_SPECIFIC(dropout);

  APPLY_SPECIFIC(err) = cudnnSetDropoutDescriptor(*dropDesc, _handle, 1.0, NULL, 0, 0);

  if (APPLY_SPECIFIC(err) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not set dropout descriptor"
                 "descriptor: %s", cudnnGetErrorString(err));
    return -1;
  }

  APPLY_SPECIFIC(err) = cudnnSetRNNDescriptor(*desc, hiddenSize, numLayers, *dropDesc,
                                        input_mode, direction, dataType);

  if (APPLY_SPECIFIC(err) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not set RNN descriptor"
                 "descriptor: %s", cudnnGetErrorString(err));
    return -1;
  }

  return 0;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(dropout) != NULL)
  cudnnDestroyDropoutDescriptor(*APPLY_SPECIFIC(dropout));

