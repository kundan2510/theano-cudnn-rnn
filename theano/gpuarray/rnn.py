class GpuRnnDesc(COp):

    """
    This Op builds a RNN descriptor for use in RNN operations.

    See the doc of :func:`dnn_rnn` for a description of the parameters

    """

    __props__ = ('hiddenSize', 'numLayers', 'mode', 'precision')

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn']

    def c_lib_dirs(self):
        return [config.dnn.library_path]

    def do_constant_folding(self, node):
        return False

    def __init__(self, hiddenSize, numLayers=1, mode='gru',
                 precision="float32", direction="uni"):
        COp.__init__(self, ["rnn_desc.c"], "APPLY_SPECIFIC(rnn_desc)")

        self.mode = mode
        self.direction = direction

        assert isinstance( hiddenSize, int ), "hiddenSize should be integer"
        assert isinstance( numLayers, int ), "numLayers should be integer"

        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        assert precision in ['float16', 'float32', 'float64']
        self.precision = precision

    def make_node(self, kern_shape):

        node = Apply(self, [],
                     [CDataType("cudnnRNNDescriptor_t",
                                freefunc="cudnnDestroyRNNDescriptor")()])
        # DebugMode cannot compare the values of CDataType variables, so by
        # default it returns False all the time. To prevent DebugMode from
        # complaining because of the MergeOptimizer, we make this variable
        # always compare to True.
        out = node.outputs[0]
        out.tag.values_eq_approx = tensor.type.values_eq_approx_always_true
        return node

    def get_op_params(self):
        if self.mode == "gru":
            activation = "CUDNN_GRU"
        elif self.mode == "lstm":
            activation = "CUDNN_LSTM"
        elif self.mode == "relu":
            activation = "CUDNN_RELU"
        elif self.mode == "tanh":
            activation = "CUDNN_TANH"
        else:
            raise raise ValueError(
                     'Invalid mode {} for RNN, it must be one of' 
                     '"relu", "tanh", "lstm" and "gru"'.format(mode)
                     )

        if self.direction == 'uni':
            direction_flag = 'CUDNN_UNIDIRECTIONAL'
        elif self.direction == 'bi':
            direction_flag = 'CUDNN_BIDIRECTIONAL'
        else:
            raise raise ValueError(
                'Invalid direction {} for RNN,'
                'it should be one of "uni" and "bi" '
                'for unidirectional and bidirectional RNN '
                'respectively'.format(direction)
                )

        if self.precision == 'float16':
            precision = 'CUDNN_DATA_HALF'
        elif self.precision == 'float32':
            precision = 'CUDNN_DATA_FLOAT'
        else:
            assert self.precision == 'float64'
            precision = 'CUDNN_DATA_DOUBLE'

        return [('HIDDEN_SIZE', self.hiddenSize),
                ('NUM_LAYERS', self.numLayers),
                ('DIRECTION', direction_flag),
                ('MODE', activation),
                ('PRECISION', precision)]

    def c_code_cache_version(self):
        return (super(GpuRnnDesc, self).c_code_cache_version(), version())

# TODO: Cache code generation



class GpuDnnRnn(DnnBase):

    """
    The forward convolution.

    Parameters
    ----------
    X : input
    W : Weights
    descr :
        The RNN descriptor.
    """

    def __init__(self):
        DnnBase.__init__(self, ["rnn_base.c", "rnn_fwd.c"],
                         "APPLY_SPECIFIC(rnn_fwd)")
        if version() < 5000:
            raise RuntimeError("cuDNN RNN requires "
                               "cuDNN v5 or more recent")

    def get_op_params(self):
        return []

    def make_node(self, X, W, output, desc, seq_len):
        ctx_name = infer_context_name(X, W, output)
        X = as_gpuarray_variable(X, ctx_name)
        W = as_gpuarray_variable(W, ctx_name)
        output = as_gpuarray_variable(output, ctx_name)
        
        if (not isinstance(desc.type, CDataType) or
                desc.type.ctype != 'cudnnRNNDescriptor_t'):
            raise TypeError('desc must be cudnnRNNDescriptor_t')

        if not isinstance(seq_len, int):
            raise TypeError('Expected an integer for seq_len but got {}'.format(seq_len))

        return Apply(self, [X, W, desc, seq_len],
                     [X.type()])

    def grad(self, inp, grads):
        # img, kerns, output, desc, alpha, beta = inp
        # top, = grads

        # top = gpu_contiguous(top)

        # d_img = gpu_dnn_conv_gradI()(kerns, top, empty_like(img), desc)
        # d_kerns = gpu_dnn_conv_gradW()(img, top, empty_like(kerns), desc)
        # d_alpha = grad_not_implemented(self, 4, alpha)
        # d_beta = grad_not_implemented(self, 5, beta)

        # return [d_img * alpha, d_kerns * alpha, top * beta,
        #         DisconnectedType()(), d_alpha, d_beta]
        raise Exception("Not Implemented Error")

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    # @staticmethod
    # def get_out_shape(ishape, kshape, border_mode, subsample):
    #     """
    #     This function computes the output shape for a convolution with
    #     the specified parameters. `ishape` and `kshape` can be symbolic
    #     or scalar.

    #     """

    #     # if ishape and/or kshape are not tuples or list, but rather symbolic
    #     # vectors, turn them into lists of symbolic scalars.
    #     if not isinstance(ishape, (list, tuple)):
    #         ishape = [ishape[i] for i in range(len(subsample) + 2)]
    #     if not isinstance(kshape, (list, tuple)):
    #         kshape = [kshape[i] for i in range(len(subsample) + 2)]

    #     return get_conv_output_shape(
    #         ishape,
    #         kshape,
    #         border_mode,
    #         subsample)

    def infer_shape(self, node, shape):
        return [shape[0][0], shape[0][1], shape[1][2]]





def dnn_rnn(X, W, hiddenSize, numLayers = 1, seq_len = 10, mode='relu', direction='uni', precision=None):
    """
    RNN using cuDNN from NVIDIA.

    Parameters
    ----------
    X
        input to the RNN layer.
        shape: (BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    W
        Weights of the RNN layer
        shape: (numLayers, 2, INPUT_DIM, OUTPUT_DIM) for 'relu' and 'tanh'
    b 
        bias for RNN layer
        shape: 
    mode
        one of 'relu', 'gru', 'tanh' and 'lstm'
    direction
        one of 'uni' and 'bi'
    precision
        one of 'as_input' or None
    hiddenSize
        integer: length of the hidden state

    """
    assert (X.ndim == 3), "Only 3-d input is allowed, got {}-d.".format(X.ndim)
    if precision is None:
        precision = theano.config.dnn.conv.precision
    if precision == 'as_input':
        precision = theano.scalar.upcast(img.dtype, kerns.dtype)
    
    X = gpu_contiguous(X)
    W = gpu_contiguous(W)
    desc = GpuRnnDesc(hiddenSize, numLayers=numLayers, mode=mode,
                 precision=precision, direction=direction)
    shape_X = [shape_i_op(i)(X) for i in range(X.ndim)]
    shape_W = [shape_i_op(i)(W) for i in range(X.ndim)]


    return GpuDnnRnn()(X, W, desc, seq_len)

