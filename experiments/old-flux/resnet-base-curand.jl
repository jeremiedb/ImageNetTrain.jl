# comment out to enable default memory pool
# ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
# @info "JULIA_CUDA_MEMORY_POOL" ENV["JULIA_CUDA_MEMORY_POOL"] 

using Images
using Random
using StatsBase: sample, shuffle
using Base.Threads: @threads, nthreads
using BSON
using CSV
using DataFrames

import Base: length, getindex
using MLUtils
import MLUtils: getobs, getobs!
using DataAugmentation

using ChainRulesCore
import ChainRulesCore: rrule

using CUDA
using Metalhead
using Flux
using Flux: update!
using ParameterSchedulers

const resnet_size = 50
const batchsize = 64


function ChainRulesCore.rrule(cfg::RuleConfig, c::Chain, x::AbstractArray)
    duo = accumulate(c.layers; init=(x, nothing)) do (input, _), layer
        out, back = rrule_via_ad(cfg, layer, input)
    end
    outs = map(first, duo)
    backs = map(last, duo)
    function un_chain(dout)
        multi = accumulate(reverse(backs); init=(nothing, dout)) do (_, delta), back
            dlayer, din = back(delta)
        end
        layergrads =
            foreach(CUDA.unsafe_free!, outs)
        foreach(CUDA.unsafe_free!, map(last, multi[1:end-1]))
        return (Tangent{Chain}(; layers=reverse(map(first, multi))), last(multi[end]))
    end
    outs[end], un_chain
end
# Could restrict this to x::CuArray... for testing instead write NaN into non-CuArrays, piratically:
CUDA.unsafe_free!(x::Array) = fill!(x, NaN)
CUDA.unsafe_free!(x::Flux.Zygote.Fill) = nothing



@info "resnet" resnet_size
@info "batchsize" batchsize
@info "nthreads" nthreads()

################################
# select device - CUDA tests
################################
CUDA.device!(0)
dev = CUDA.device()
# attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)
# release_threshold = UInt64(1e9)
# Int(attribute(UInt64, CuMemoryPool(dev), CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD))
# Int(attribute(UInt64, memory_pool(dev), CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD))
# # attribute!(CuMemoryPool(dev), CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt(release_threshold))
# attribute!(memory_pool(dev), CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt(release_threshold))
# Int(attribute(UInt64, memory_pool(dev), CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD))
# Int(CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD)

#set model input image size
const im_size = (224, 224)

Random.seed!(123)
nobs = 1281144 ÷ 1000
nbatch = cld(nobs, batchsize)
@info "nobs" nobs
@info "nbatch" nbatch

function loss(m, x, y)
    Flux.Losses.logitcrossentropy(m(x), y)
end

const m_device = gpu

function train_epoch!(m, ps, opt, loss; nbatch)
    x, y = CUDA.rand(im_size..., 3, batchsize), Flux.onehotbatch(rand(1:1000, batchsize), 1:1000) |> m_device
    for batch in 1:nbatch
        # GC.gc(true)
        # CUDA.reclaim()
        # x, y = CUDA.rand(im_size..., 3, batchsize), Flux.onehotbatch(rand(1:1000, batchsize), 1:1000) |> m_device
        grads = gradient(ps) do
            loss(m, x, y)
        end
        update!(opt, ps, grads)
    end
end

@info "loading model / optimizer"
m = ResNet(resnet_size, nclasses=1000) |> m_device;
ps = Flux.params(m);
opt = Flux.Optimise.Nesterov(1.0f-5)

function train_loop(iter_start, iter_end)
    for i in iter_start:iter_end
        GC.gc(true)
        CUDA.reclaim()
        @time train_epoch!(m, ps, opt, loss; nbatch)
    end
end

train_loop(1, 3)
