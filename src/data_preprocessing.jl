using CSV, DataFrames, Flux, GraphNeuralNetworks

function load_data(filepath::String)
    df = CSV.read(filepath, DataFrame)
    return df
end

function preprocess_data(df::DataFrame)
    # Add code to process data, create graph structure, etc.
    return graph_data
end

train_data = preprocess_data(load_data("data/raw_data/graph_data.csv"))
