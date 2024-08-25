using Flux, GraphNeuralNetworks

# Evaluation function for metabolic pathway model
function evaluate_model(model, graph, features, labels)
    gs = gnn_data(graph, features, labels)
    predictions = model(gs)
    accuracy = mean(argmax(predictions, dims=2) .== labels)
    println("Test Accuracy: $accuracy")
end

evaluate_model(trained_model, graph, features, labels)
