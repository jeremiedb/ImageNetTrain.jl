using Images
using Random
using StatsBase: sample, shuffle
using Base.Threads: @threads, nthreads
using BSON
using CSV
using DataFrames

import Base: length, getindex
using DataAugmentation

using CUDA
using Metalhead
using Flux
using Flux: MLUtils, DataLoader, Optimisers

const resnet_size = 34
const iter = 90
const batchsize = 128
#set model input image size
const im_size_pre = (256, 256)
const im_size = (224, 224)

const m_device = gpu
const results_path = "results/resnet"

@info "resnet" resnet_size
@info "batchsize" batchsize
@info "nthreads" nthreads()

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

# val image container
struct ValContainer{T<:Vector,S<:Vector}
    img::T
    key::S
end

length(data::ValContainer) = length(data.img)
tfm_val = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), CenterCrop(im_size))
# tfm_val = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))
# tfm_val = DataAugmentation.compose(CenterCrop(im_size))

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

# tfm_val = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), CenterCrop(im_size), ImageToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
# function getindex(data::ValContainer, idx::Int)
#     path = data.img[idx]
#     y = key_to_idx[data.key[idx]]
#     img = Images.load(path)
#     x = collect(itemdata(apply(tfm_val, Image(img))))
#     # x = collect(itemdata(x))
#     return (x, y)
# end

# # set data loaders
# deval = DataLoader(ValContainer(imgs_val, key_val); batchsize, partial=false, parallel=true, collate=true)

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

# m = ResNet(resnet_size; pretrain = true) |> m_device;
m = BSON.load(joinpath(results_path, "resnet$(resnet_size)-optim-Nesterov-B-$iter.bson"), @__MODULE__)[:model] |> m_device;

CUDA.allowscalar(false)
metric = eval_f(m, deval)
@info metric

# tfm_val = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), CenterCrop(im_size))
# 0.7287059294871795

# tfm_val = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))
# 0.7210336538461538

# tfm_val = DataAugmentation.compose(CenterCrop(im_size))
# 0.671113782051282