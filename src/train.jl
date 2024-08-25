using Flux, GraphNeuralNetworks

# Training loop adjusted for metabolic pathway data
function train_model(model, graph, features, labels, epochs::Int, learning_rate::Float64)
    opt = ADAM(learning_rate)
    for epoch in 1:epochs
        gs = gnn_data(graph, features, labels)
        Flux.train!(loss, model, gs, opt)
        println("Epoch $epoch completed")
    end
    return model
end

labels = rand(1:5, length(features))  # Example labels
trained_model = train_model(model, graph, features, labels, 100, 0.001)
