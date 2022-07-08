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
using Optimisers

resnet_size = 34
batchsize = 128

@info "resnet" resnet_size
@info "nthreads" nthreads()

# select device
# CUDA.device!(0)

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
# num_obs = 1_000
@info "num_obs" num_obs

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

# set data loaders
dtrain = DataLoader(ImageContainer(imgs[idtrain]); batchsize, partial=false, parallel=true, collate=true)

function loop_data_cpu(dtrain)
    iter = 1
    for (x, y,) in dtrain
        @info "iter" iter
        # sum(x)
        iter += 1
    end
end

function loop_data_gpu(dtrain)
    for (x, y,) in dtrain
        sum(x |> gpu)
    end
end

function loop_data_cuiter(dtrain)
    for (iter, (x, y,)) in enumerate(CuIterator(dtrain))
        sum(x)
    end
end

@info "start cpu loop"
@time loop_data_cpu(dtrain)
# @time loop_data_cpu(dtrain)
# @info "start gpu loop"
# CUDA.@time loop_data_gpu(dtrain)
# CUDA.@time loop_data_gpu(dtrain)
# @info "start cuiter loop"
# CUDA.@time loop_data_cuiter(dtrain)
# CUDA.@time loop_data_cuiter(dtrain)
