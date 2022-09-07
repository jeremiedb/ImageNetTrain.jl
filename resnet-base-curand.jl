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

using CUDA
using Metalhead
using Flux
using Flux: update!
using ParameterSchedulers

const resnet_size = 50
const batchsize = 32

@info "resnet" resnet_size
@info "nthreads" nthreads()


# select device
CUDA.device!(0)
release_threshold = 0
attribute!(memory_pool(device()), CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt(release_threshold))

#set model input image size
const im_size = (224, 224)

Random.seed!(123)
nobs = 1281144 ÷ 1000
nbatch = nobs ÷ batchsize
@info "nobs" nobs
@info "nbatch" nbatch

function loss(m, x, y)
    Flux.Losses.logitcrossentropy(m(x), y)
end

const m_device = gpu

function train_epoch!(m, θ, opt, loss; nbatch)
    x, y = CUDA.rand(im_size..., 3, batchsize), Flux.onehotbatch(rand(1:1000, batchsize), 1:1000) |> m_device
    for batch in 1:nbatch
        # x, y = CUDA.rand(im_size..., 3, batchsize), Flux.onehotbatch(rand(1:1000, batchsize), 1:1000) |> m_device
        grads = gradient(θ) do
            loss(m, x, y)
        end
        update!(opt, θ, grads)
    end
end

@info "loading model / optimizer"
m = ResNet(resnet_size, nclasses=1000) |> m_device;
θ = Flux.params(m);
opt = Flux.Optimise.Nesterov(1.0f-5)

function train_loop(iter_start, iter_end)
    for i in iter_start:iter_end
        GC.gc(true)
        CUDA.reclaim()
        @time train_epoch!(m, θ, opt, loss; nbatch)
    end
end

train_loop(1, 3)
