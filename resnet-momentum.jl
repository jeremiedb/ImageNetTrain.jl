using Images
using Random
using StatsBase: sample, shuffle
using Base.Threads: @threads, nthreads
using BSON
using CSV
using DataFrames

import Base: length, getindex
using DataAugmentation

using ChainRulesCore
import ChainRulesCore: rrule

using CUDA
using Metalhead
using Flux
using Flux: MLUtils, DataLoader, Optimisers
using ParameterSchedulers

const resnet_size = 34
const batchsize = 128
#set model input image size
const im_size_pre = (256, 256)
const im_size = (224, 224)

const m_device = gpu
const results_path = "results/resnet"

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
val_img_path = "data/ILSVRC/Data/CLS-LOC/val_ori/"
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

tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size), Maybe(FlipX()), AdjustContrast(0.3), AdjustBrightness(0.3))
# tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size), Maybe(FlipX()), Maybe(FlipY()), AdjustContrast(0.1), AdjustBrightness(0.1))
# tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size), Maybe(FlipX()), Maybe(FlipY()), AdjustContrast(0.3), AdjustBrightness(0.3))
# tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size), Maybe(FlipX()))
# tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size))
# tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), CenterCrop(im_size))

function getindex(data::ImageContainer, idx::Int)
    path = data.img[idx]
    y = train_path_to_idx(path)
    img = Images.load(path)
    img = apply(tfm_train, Image(img))
    x = collect(channelview(float32.(RGB.(itemdata(img)))))
    mu = Float32.([0.485, 0.456, 0.406])
    sigma = Float32.([0.229, 0.224, 0.225])
    x = permutedims((x .- mu) ./ sigma, (3, 2, 1))
    return (x, y)
    # return (x, Flux.onehotbatch(y, 1:1000))
end

# val image container
struct ValContainer{T<:Vector,S<:Vector}
    img::T
    key::S
end

length(data::ValContainer) = length(data.img)
tfm_val = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), CenterCrop(im_size))

function getindex(data::ValContainer, idx::Int)
    path = data.img[idx]
    y = key_to_idx[data.key[idx]]
    img = Images.load(path)
    img = apply(tfm_val, Image(img))
    x = collect(channelview(float32.(RGB.(itemdata(img)))))
    mu = Float32.([0.485, 0.456, 0.406])
    sigma = Float32.([0.229, 0.224, 0.225])
    x = permutedims((x .- mu) ./ sigma, (3, 2, 1))
    return (x, y)
end

# set data loaders
dtrain = DataLoader(ImageContainer(imgs[idtrain]); batchsize, partial=false, parallel=true, collate=true)
deval = DataLoader(ValContainer(imgs_val, key_val); batchsize, partial=false, parallel=true, collate=true)

# loss
function loss(m, x, y)
    Flux.Losses.logitcrossentropy(m(x), Flux.onehotbatch(y, 1:1000))
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

function train_loop(iter_start, iter_end)

    if iter_start == 1
        m = ResNet(resnet_size, nclasses=1000) |> m_device
        rule = Optimisers.OptimiserChain(Optimisers.WeightDecay(3.0f-5), Optimisers.Momentum(1.0f-1, 0.875f0))
        opts = Flux.setup(rule, m)
    else
        init = iter_start - 1
        m = BSON.load(joinpath(results_path, "resnet$(resnet_size)-Momentum-A-$init.bson"), @__MODULE__)[:model] |> m_device
        opts = BSON.load(joinpath(results_path, "resnet$(resnet_size)-Momentum-A-$init.bson"), @__MODULE__)[:opts] |> m_device
    end

    for i in iter_start:iter_end
        @info "iter: " i
        if i == iter_start
            metric = eval_f(m, deval)
            @info metric
        end
        if i == 1
            Optimisers.adjust!(opts, 1.0f-3)
            @info "optim adjustment"
        elseif i == 2
            Optimisers.adjust!(opts, 1.0f-2)
            @info "optim adjustment"
        elseif i == 3
            Optimisers.adjust!(opts, 1.0f-1)
            @info "optim adjustment"
        elseif i == 16
            Optimisers.adjust!(opts, 1.0f-2)
            @info "optim adjustment"
        elseif i == 21
            Optimisers.adjust!(opts, 5.0f-2)
            @info "optim adjustment"
        elseif i == 26
            Optimisers.adjust!(opts, 5.0f-3)
            @info "optim adjustment"
        elseif i == 31
            Optimisers.adjust!(opts, 1.0f-2)
            @info "optim adjustment"
        elseif i == 36
            Optimisers.adjust!(opts, 1.0f-3)
            @info "optim adjustment"
        elseif i == 41
            Optimisers.adjust!(opts, 5.0f-3)
            @info "optim adjustment"
        elseif i == 46
            Optimisers.adjust!(opts, 1.0f-3)
            @info "optim adjustment"
        end
        @time train_epoch!(m, opts, loss; dtrain=dtrain)
        metric = eval_f(m, deval)
        @info metric
        BSON.bson(joinpath(results_path, "resnet$(resnet_size)-Momentum-A-$i.bson"), Dict(:model => m |> cpu, :opts => opts |> cpu))
    end
end

@info "Start training"
train_loop(1, 50)