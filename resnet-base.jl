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
const batchsize = 32

@info "resnet" resnet_size
@info "nthreads" nthreads()

# select device
CUDA.device!(0)

#set model input image size
const im_size_pre = (256, 256)
const im_size = (224, 224)

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

# train eval split
updates_per_epoch = Int(floor(num_obs / batchsize))
idtrain = shuffle(1:length(imgs))[1:num_obs]

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

tfm_train = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), RandomCrop(im_size))

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

function train_epoch!(m, θ, opt, loss; dtrain)
    for (x, y) in dtrain
        grads = gradient(θ) do
            loss(m, x |> gpu, Flux.onehotbatch(y, 1:1000) |> gpu)
        end
        update!(opt, θ, grads)
    end
end

m_device = gpu
m = ResNet(50, nclasses=1000) |> m_device;
#@info "loading model"
#m = BSON.load("results/model-opt-iter-A-22.bson")[:model] |> m_device;
#@info "loading optmiser"
#opt = BSON.load("results/model-opt-iter-A-22.bson")[:opt] |> m_device;

θ = Flux.params(m);
# θ = θ[length(θ)-1:length(θ)];

# opt = Flux.Optimise.Adam(1.0f-3)
opt = Flux.Optimise.Nesterov(1.0f-2)
#opt = Adam(1e-2)
#s = ParameterSchedulers.Sequence(1f-4 => 1 * updates_per_epoch, 3f-4 => 1 * updates_per_epoch, 1f-3 => 14 * updates_per_epoch,
#    1f-4 => 12 * updates_per_epoch, 3f-4 => 4 * updates_per_epoch, 3f-5 => 12 * updates_per_epoch, 1f-5 => 4 * updates_per_epoch)
#opt = ParameterSchedulers.Scheduler(s, Adam())

results_path = "results"

function train_loop(epochs)
    for i in 1:epochs
        @info "iter: " i
        @info "opt.eta" opt.eta
        if i == 1
            @time metric = eval_f(m, deval)
            @info metric
        end
        @time train_epoch!(m, θ, opt, loss; dtrain=dtrain)
        metric = eval_f(m, deval)
        @info metric
        BSON.bson(joinpath(results_path, "resnet$(resnet_size)-A-$i.bson"), Dict(:model => m |> cpu, :opt => opt |> cpu))
        if i == 1
            opt.eta = 1f-2
        end
        if i % 20 == 0
            opt.eta /= 10
        end
    end
end

@time train_loop(1)