module JuliaBot

using Flux: Chain, Dense, relu, softmax
using Flux.Data: DataLoader
using Flux: @epochs

export train_model, predict

struct AIModel
    model::Chain
end

function AIModel()
    return AIModel(Chain(
        Dense(784, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 10),
        softmax
    ))
end

function train_model(data_loader::DataLoader, model::AIModel, epochs::Int)
    loss(x, y) = Flux.Losses.crossentropy(model.model(x), y)
    opt = ADAM()
    @epochs epochs Flux.train!(loss, params(model.model), data_loader, opt)
end

function predict(model::AIModel, input)
    return model.model(input)
end

end # module JuliaBot
