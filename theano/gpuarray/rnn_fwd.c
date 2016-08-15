#section support_code_struct

int
APPLY_SPECIFIC(rnn_fwd)(PyGpuArrayObject *X, PyGpuArrayObject *W,
                         PyGpuArrayObject *output,
                         cudnnRNNDescriptor_t desc,
                         int seqLength
                         ) {
	cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

	size_t worksize;
    gpudata *workspace;
    err = cudnnGetRNNWorkspaceSize(APPLY_SPECIFIC(_handle),
                                   desc,
                                   seqLength,
                                   APPLY_SPECIFIC(X),
                                   &worksize);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error getting worksize: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }

    if (worksize != 0) {
      workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
      if (workspace == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Could not allocate working memory");
        cuda_exit(c->ctx);
        return 1;
      }
    }


    size_t reserveSize;
    gpudata *reserveSpace;

    err = cudnnGetRNNTrainingReserveSize(APPLY_SPECIFIC(_handle),
                                   desc,
                                   seqLength,
                                   APPLY_SPECIFIC(X),
                                   &reserveSize);

    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error getting reserveSize for RNN: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }

    if (reserveSize != 0) {
      reserveSpace = gpudata_alloc(c->ctx, reserveSpace, NULL, 0, NULL);
      if (reserveSpace == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Could not allocate reserveSpace memory");
        cuda_exit(c->ctx);
        return 1;
      }
    }

    err = cudnnRNNForwardTraining( APPLY_SPECIFIC(_handle),
			desc,
			seqLength,
			APPLY_SPECIFIC(X), PyGpuArray_DEV_DATA(X),
			*APPLY_SPECIFIC(hxDesc), NULL,
			*APPLY_SPECIFIC(cxDesc), NULL,
			*APPLY_SPECIFIC(W), PyGpuArray_DEV_DATA(W),
			APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(output),
			*APPLY_SPECIFIC(hyDesc), NULL,
			*APPLY_SPECIFIC(cyDesc), NULL,
			worksize == 0 ? NULL : *(void **)workspace, worksize,
			reserveSize == 0 ? NULL : *(void **)reserveSpace, reserveSize,
		)
	return 0;
}