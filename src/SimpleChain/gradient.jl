using SimpleChains
using Parameters: @unpack
using SimpleChains: has_loss


function valgrad_input!(g, c::SimpleChain, arg::AbstractArray{T0}, params::AbstractVector{T1}) where {T0,T1}
    SimpleChains.verify_arg(c, arg)
    #@assert has_loss(c)
    @unpack layers = c
    parg = SimpleChains.maybe_static_size_arg(c.inputdim, arg)
    num_bytes = SimpleChains.required_bytes(Val{promote_type(T0, T1)}(), layers, SimpleChains.static_size(parg), static(sizeof(T0)))
    GC.@preserve arg SimpleChains.with_memory(unsafe_valgrad_input!, c, num_bytes, g, params, parg)
end

function unsafe_valgrad_input!(
  c::SimpleChains.Chain,
  pu::Ptr{UInt8},
  g::AbstractArray,
  params,
  arg
)#(pg, c::SimpleChain, num_bytes::Int, g, params, parg)
    pu = SimpleChains.unsafe_wrap_metadata(pg, num_bytes)
    pu2 = pu + sizeof(parg)
    arg_ptr = pointer_from_objref(parg)
    chain_valgrad!(pg, arg_ptr, c.layers, pointer_from_objref(params[1]), pu2)
    pullback!(pg, c.loss, g, c.out, arg_ptr, pu, pu2)
end



c = SimpleChain(
    static(2),
    TurboDense(tanh, 32),
    TurboDense(tanh, 16),
    TurboDense(identity, 1)
)
batch_size = 64
X = rand(Float32, 2, batch_size)
Y = rand(Bool, batch_size)

params = SimpleChains.init_params(c)
c_loss = SimpleChains.add_loss(c, SquaredLoss(Y));
g = zeros(2)
valgrad_input!(g, c, X, params)
