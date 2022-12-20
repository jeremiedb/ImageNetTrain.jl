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
using Optimisers

using ChainRulesCore
import ChainRulesCore: rrule

using CUDA
using Metalhead
using Flux
using Flux: update!
using ParameterSchedulers

const resnet_size = 50
const batchsize = 64

@info "resnet" resnet_size
@info "batchsize" batchsize
@info "nthreads" nthreads()

################################
# select device - CUDA tests
################################
CUDA.device!(0)
# dev = CUDA.device()
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

CUDA.allowscalar(true)
function train_epoch!(m, opts, loss; nbatch)
    x, y = CUDA.rand(im_size..., 3, batchsize), Flux.onehotbatch(rand(1:1000, batchsize), 1:1000) |> m_device
    # x, y = CUDA.rand(im_size..., 3, batchsize), rand(1:1000, batchsize) |> m_device
    for batch in 1:nbatch
        # GC.gc(true)
        # CUDA.reclaim()
        grads = gradient((model) -> loss(model, x, y), m)[1]
        Optimisers.update!(opts, m, grads)
    end
end

@info "loading model / optimizer"

function train_loop(iter_start, iter_end)
    m = ResNet(resnet_size, nclasses=1000) |> m_device
    rule = Optimisers.Nesterov(1.0f-3)
    opts = Optimisers.setup(rule, m)
    for i in iter_start:iter_end
        @time train_epoch!(m, opts, loss; nbatch)
    end
end

train_loop(1, 3)
