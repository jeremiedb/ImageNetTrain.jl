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

batchsize = 128
@info "nthreads" nthreads()

#set model input image size
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

# train eval split
updates_per_epoch = Int(floor(num_obs / batchsize))
idtrain = shuffle(1:length(imgs))[1:num_obs]

# train image container
struct ImageContainer{T<:Vector}
    img::T
end

length(data::ImageContainer) = length(data.img)

tfm = DataAugmentation.compose(CenterCrop(im_size))

function getobs(data::ImageContainer, idx::Int)
    path = data.img[idx]
    img = Images.load(path)
    img = apply(tfm, Image(Images.load(path)))
    x = permutedims(channelview(RGB.(itemdata(img))), (3, 2, 1))
    return Float32.(x)
end

function getobs!(buffer, data::ImageContainer, idx::Int)
    path = data.img[idx]
    img = apply(tfm, Image(Images.load(path)))
    buffer .= permutedims(channelview(RGB.(itemdata(img))), (3, 2, 1))
    return buffer
end

function getindex(data::ImageContainer, idx::AbstractVector)
    x = zeros(Float32, im_size..., 3, batchsize)
    @threads for i in 1:length(idx)
        path = data.img[idx[i]]
        # img = Images.load(path)
        img = apply(tfm, Image(Images.load(path)))
        x1 = permutedims(channelview(RGB.(itemdata(img))), (3, 2, 1))
        x[:, :, :, i] = x1
    end
    return x
end

# set data loaders
dtrain = DataLoader(ImageContainer(imgs[idtrain]); batchsize, partial=false, parallel=true, collate=nothing, buffer=false)

function loop_data_cpu(dtrain)
    iter = 1
    for (x, y,) in dtrain
        @info "iter" iter
        # sum(x)
        iter += 1
    end
end

@info "start cpu loop"
@time loop_data_cpu(dtrain)