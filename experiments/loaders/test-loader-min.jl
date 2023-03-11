using Images
using Random
using StatsBase: sample, shuffle
using Base.Threads: @threads, nthreads
using BSON
using CSV
using DataFrames
using CUDA

import Base: length, getindex
using Flux: MLUtils, DataLoader, update!
using DataAugmentation
using Metalhead
using ParameterSchedulers
using Optimisers

batchsize = 128
@info "nthreads" nthreads()

#set model input image size
const im_size_pre = (256, 256)
const im_size = (224, 224)

# list images
train_img_path = joinpath(@__DIR__, "../..", "data/ILSVRC/Data/CLS-LOC/train")
imgs = vcat([readdir(dir, join=true) for dir in readdir(train_img_path, join=true)]...)
# remove CMY / png images: https://github.com/cytsai/ilsvrc-cmyk-image-list
include("../../utils.jl");
imgs_delete = joinpath.(train_img_path, first.(rsplit.(imgs_delete, "_", limit=2)), imgs_delete)
setdiff!(imgs, imgs_delete)

Random.seed!(123)
# num_obs = length(imgs)
num_obs = 1_000
@info "num_obs" num_obs

# train eval split
updates_per_epoch = Int(floor(num_obs / batchsize))
idtrain = shuffle(1:length(imgs))[1:num_obs]

# train image container
struct ImageContainer{T<:Vector}
    img::T
end

length(data::ImageContainer) = length(data.img)

tfm = DataAugmentation.compose(ScaleKeepAspect(im_size_pre), CenterCrop(im_size))

function getindex(data::ImageContainer, idx::Int)
    path = data.img[idx]
    img = Images.load(path)
    img = apply(tfm, Image(Images.load(path)))
    # x = permutedims(collect(channelview(float32.(itemdata(img)))), (3, 2, 1))
    # x = permutedims(collect(channelview(RGB.(itemdata(img)))), (3, 2, 1))
    x = collect(channelview(float32.(RGB.(itemdata(img)))))
    # x = collect(channelview(RGB.(itemdata(img))))
    return permutedims(x, (3,2,1))
end

# function getobs!(buffer, data::ImageContainer, idx::Int)
#     path = data.img[idx]
#     img = apply(tfm, Image(Images.load(path)))
#     buffer .= permutedims(channelview(RGB.(itemdata(img))), (3, 2, 1))
#     return buffer
# end

# function getindex(data::ImageContainer, idx::AbstractVector)
#     x = zeros(Float32, im_size..., 3, batchsize)
#     @threads for i in 1:length(idx)
#         path = data.img[idx[i]]
#         # img = Images.load(path)
#         img = apply(tfm, Image(Images.load(path)))
#         x1 = permutedims(channelview(RGB.(itemdata(img))), (3, 2, 1))
#         x[:, :, :, i] = x1
#     end
#     return x
# end

# container = ImageContainer(imgs[idtrain]);
# getobs(container, 1)

# set data loaders
dtrain = DataLoader(ImageContainer(imgs[idtrain]); batchsize, partial=true, parallel=true, collate=true, buffer=false);

function loop_data_cpu(dtrain)
    iter = 1
    for x in dtrain
        @info "iter" iter
        @info "size" size(x)
        # sum(x)
        iter += 1
    end
end

@info "start cpu loop"
@time loop_data_cpu(dtrain)