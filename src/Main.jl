include("data_preprocessing.jl")
include("gnn_model.jl")
include("train.jl")
include("evaluate.jl")

graph, features = load_and_preprocess_data("data/raw_data/metabolic_pathway.csv")
model = create_gnn_model(10, 5)
trained_model = train_model(model, graph, features, labels, 100, 0.001)
evaluate_model(trained_model, graph, features, labels)
