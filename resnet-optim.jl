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
using Metalhead: PartialFunctions
using Flux
using Flux: update!
using ParameterSchedulers
using Optimisers

const resnet_size = 50
const batchsize = 64
#set model input image size
const im_size_pre = (256, 256)
const im_size = (224, 224)

@info "resnet" resnet_size
@info "batchsize" batchsize
@info "nthreads" nthreads()

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

# select device
CUDA.device!(0)

# list images
train_img_path = joinpath(@__DIR__, "data/ILSVRC/Data/CLS-LOC/train")
imgs = vcat([readdir(dir, join=true) for dir in readdir(train_img_path, join=true)]...)
# remove CMY / png images: https://github.com/cytsai/ilsvrc-cmyk-image-list
include("utils.jl");
imgs_delete = joinpath.(train_img_path, first.(rsplit.(imgs_delete, "_", limit=2)), imgs_delete)
setdiff!(imgs, imgs_delete)

Random.seed!(123)
num_obs = length(imgs)
# num_obs = 10_000
@info "num_obs" num_obs
updates_per_epoch = Int(floor(num_obs / batchsize))
idtrain = shuffle(1:length(imgs))[1:num_obs]

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

# tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))
tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size), Maybe(FlipX()))
# tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size), Maybe(FlipX()), AdjustContrast(0.2), AdjustBrightness(0.2))

function getindex(data::ImageContainer, idx::Int)
    path = data.img[idx]
    y = train_path_to_idx(path)
    x = Images.load(path)
    x = apply(tfm_train, Image(x))
    x = permutedims(channelview(RGB.(itemdata(x))), (3, 2, 1))
    x = Float32.(x)
    # return (x, y)
    return (x, Flux.onehotbatch(y, 1:1000))
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

function train_epoch!(m, opts, loss; dtrain)
    for (batch, (x, y)) in enumerate(CuIterator(dtrain))
        grads = gradient((model) -> loss(model, x, y), m)[1]
        Optimisers.update!(opts, m, grads)
    end
    return nothing
end

const m_device = gpu
const results_path = "results"

function train_loop(iter_start, iter_end)

    if iter_start == 1
        m = ResNet(resnet_size, nclasses=1000) |> m_device
        rule = Optimisers.Nesterov(1.0f-2)
        # rule = Optimisers.Adam(1f-3)
        opts = Optimisers.setup(rule, m)
    else
        # @error "model recovery not yet supported "
        iter_init = iter_start - 1
        m = BSON.load("results/resnet$(resnet_size)-optim-Nesterov-A-$(iter_init).bson", @__MODULE__)[:model] |> m_device
        opts = BSON.load("results/resnet$(resnet_size)-optim-Nesterov-A-$(iter_init).bson", @__MODULE__)[:opts] |> m_device
    end

    for i in iter_start:iter_end
        @info "iter: " i
        if i == iter_start
            metric = eval_f(m, deval)
            @info metric
        end
        @time train_epoch!(m, opts, loss; dtrain=dtrain)
        metric = eval_f(m, deval)
        @info metric
        BSON.bson(joinpath(results_path, "resnet$(resnet_size)-optim-Nesterov-A-$i.bson"), Dict(:model => m |> cpu, :opts => opts |> cpu))
        if i > 20
            Optimisers.adjust!(opts, 1e-3)
        elseif i > 40
            Optimisers.adjust!(opts, 1e-4)
        elseif i > 60
            Optimisers.adjust!(opts, 3e-5)
        end
    end
end

@info "Start training"
train_loop(11, 24)