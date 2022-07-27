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

const resnet_size = 34
const batchsize = 64

@info "resnet" resnet_size
@info "nthreads" nthreads()

# select device
CUDA.device!(0)

#set model input image size
const im_size_pre = (256, 256)
const im_size = (224, 224)

# unsafe_free OneHotArrays
# CUDA.unsafe_free!(x::Flux.OneHotArray) = CUDA.unsafe_free!(x.indices)

# list images
train_img_path = joinpath(@__DIR__, "data/ILSVRC/Data/CLS-LOC/train")
imgs = vcat([readdir(dir, join=true) for dir in readdir(train_img_path, join=true)]...)
# remove CMY / png images: https://github.com/cytsai/ilsvrc-cmyk-image-list
include("utils.jl");
imgs_delete = joinpath.(train_img_path, first.(rsplit.(imgs_delete, "_", limit=2)), imgs_delete)
setdiff!(imgs, imgs_delete)

Random.seed!(123)
nobs = length(imgs)
@info "nobs" nobs
updates_per_epoch = Int(floor(nobs / batchsize))
idtrain = shuffle(1:length(imgs))[1:nobs]

# list val images
val_img_path = "data/ILSVRC/Data/CLS-LOC/val/"
val_mapping = CSV.read("data/LOC_val_solution.csv", DataFrame)
transform!(val_mapping, :PredictionString => ByRow(x -> x[1:9]) => :label_key)
imgs_val = val_mapping[:, :ImageId]
imgs_val = val_img_path .* imgs_val .* ".JPEG"
key_val = val_mapping[:, :label_key]

# labels mapping
key_to_idx = Dict{String,Int}()
key_to_name = Dict{String,String}()
idx_to_name = Dict{Int,String}()
for (idx, line) in enumerate(readlines("data/LOC_synset_mapping.txt"))
    key = replace(line, r"(n\d+)\s(.+)" => s"\1")
    name = replace(line, r"(n\d+)\s(\w+),?(.+)" => s"\2")
    push!(key_to_idx, key => idx)
    push!(key_to_name, key => name)
    push!(idx_to_name, idx => name)
end

# train image container
struct ImageContainer{T<:Vector}
    img::T
end

length(data::ImageContainer) = length(data.img)

function train_path_to_idx(path)
    key = replace(path, r".+/(n\d+)/.+_\d+\.JPEG" => s"\1")
    idx = key_to_idx[key]
    return idx
end

tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size), Maybe(FlipX()), AdjustContrast(0.2), AdjustBrightness(0.2))
# tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))

function getindex(data::ImageContainer, idx::Int)
    path = data.img[idx]
    y = train_path_to_idx(path)
    x = Images.load(path)
    x = apply(tfm_train, Image(x))
    x = permutedims(channelview(RGB.(itemdata(x))), (3, 2, 1))
    x = Float32.(x)
    return (x, y)
end

# val image container
struct ValContainer{T<:Vector,S<:Vector}
    img::T
    key::S
end

length(data::ValContainer) = length(data.img)

tfm_val = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))

function getindex(data::ValContainer, idx::Int)
    path = data.img[idx]
    y = key_to_idx[data.key[idx]]
    x = Images.load(path)
    x = apply(tfm_val, Image(x))
    x = permutedims(channelview(RGB.(itemdata(x))), (3, 2, 1))
    x = Float32.(x)
    return (x, y)
end

# set data loaders
dtrain = DataLoader(ImageContainer(imgs[idtrain]); batchsize, partial=false, parallel=true, collate=true)
deval = DataLoader(ValContainer(imgs_val, key_val); batchsize, partial=false, parallel=true, collate=true)

#@info "iterating on eval data"
#for (x,y) in deval
#   println(size(y))
#   println(typeof(y))
#end

function loss(m, x, y)
    Flux.Losses.logitcrossentropy(m(x), y)
end

function eval_f(m, data)
    good = 0
    count = 0
    for (x, y) in data
        good += sum(Flux.onecold(cpu(m(x |> gpu))) .== y)
        count += length(y)
    end
    acc = good / count
    return acc
end

# function train_epoch!(m, θ, opt, loss; dtrain)
#     for (x, y) in dtrain
#         grads = gradient(θ) do
#             loss(m, x |> gpu, Flux.onehotbatch(y, 1:1000) |> gpu)
#         end
#         update!(opt, θ, grads)
#     end
# end

function train_epoch!(m, θ, opt, loss; dtrain)
    for (batch, (x, y)) in enumerate(CuIterator(dtrain))
        grads = gradient(θ) do
            loss(m, x, Flux.onehotbatch(y, 1:1000))
        end
        update!(opt, θ, grads)
        batch % 200 == 0 && GC.gc(true)
    end
end

const m_device = gpu

# @info "loading model / optimizer"
# m = ResNet(resnet_size, nclasses=1000) |> m_device;
# θ = Flux.params(m);

# opt = Flux.Optimise.Nesterov(1.0f-3)
# opt = Flux.Optimise.NAdam(1.0f-5)

results_path = "results"

# @time metric = eval_f(m, deval)
# @info "eval metric" metric

function train_loop(iter_start, iter_end)

    iter_init = iter_start - 1
    m = BSON.load("results/resnet$(resnet_size)-base-NAdam-A-$(iter_init).bson")[:model] |> m_device
    opt = BSON.load("results/resnet$(resnet_size)-base-NAdam-A-$(iter_init).bson")[:opt] |> m_device
    θ = Flux.params(m)

    for i in iter_start:iter_end
        GC.gc(true)
        CUDA.reclaim()

        if i == 1
            opt.eta = 1.0f-5
        elseif i == 2
            opt.eta = 1.0f-4
        elseif i == 3
            opt.eta = 1.0f-3
        elseif i == 31
            opt.eta = 1.0f-4
        elseif i == 41
            opt.eta = 3.0f-5
        end

        @info "iter: " i
        @info "opt.eta" opt.eta
        @time train_epoch!(m, θ, opt, loss; dtrain=dtrain)
        @info "training epoch $i completed"
        metric = eval_f(m, deval)
        @info "eval metric" metric
        BSON.bson(joinpath(results_path, "resnet$(resnet_size)-base-NAdam-A-$i.bson"), Dict(:model => m |> cpu, :opt => opt |> cpu))
    end
end

# @time train_loop(1, 16)
# @time train_loop(11, 20)
# @time train_loop(21, 30)
@time train_loop(31, 42)
# @time train_loop(33, 48)
# @time train_loop(25, 36)

function metric_loop(iter_start, iter_end)
    for i in iter_start:iter_end
        # Some Housekeeping
        GC.gc(true)
        CUDA.reclaim()

        m = BSON.load("results/resnet$(resnet_size)-base-NAdam-A-$i.bson")[:model] |> m_device
        opt = BSON.load("results/resnet$(resnet_size)-base-NAdam-A-$i.bson")[:opt] |> m_device

        @info "iter: " i
        @info "opt.eta" opt.eta
        metric = eval_f(m, deval)
        @info "eval metric" metric
    end
end

# @time metric_loop(1, 20)