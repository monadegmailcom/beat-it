#include "libtorch_util.h"

#include <map>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <boost/json.hpp>
#include <boost/json/src.hpp> // use header only version

using namespace std;

namespace libtorch {

torch::Device get_device()
{
    // Check for available hardware backends, preferring MPS on Apple Silicon, 
    // then CUDA.
    if (torch::mps::is_available())
        return torch::kMPS;
    else if (torch::cuda::is_available())
        return torch::kCUDA;
    else
        return torch::kCPU;
}

// Helper function to safely get a value from a JSON object.
// Throws a descriptive error if the key is not found.
template <typename T>
T get_required_value(const boost::json::object& obj, string const& key)
{
    if (!obj.contains(key))
        throw invalid_argument(
            "Required key not found in self_play_config: '" + key + "'");
    return boost::json::value_to<T>(obj.at(key));
}

template< typename T >
T get_value_with_default(
    const boost::json::object& obj, string const& key,
    T const& default_value)
{
    if (!obj.contains(key))
        return default_value;
    return boost::json::value_to<T>(obj.at(key));
}

pair< unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model(
    const char* model_path, torch::Device device )
{
    // Read the entire model file into a string buffer.
    ifstream model_file(model_path, ios::binary);
    if (!model_file)
        throw invalid_argument(
            "Failed to open model file: " + string(model_path));

    stringstream model_buffer;
    model_buffer << model_file.rdbuf();
    string model_data = model_buffer.str();

    // Use `unzip` via `popen` to robustly extract the metadata.json.
    // This is more reliable than relying on libtorch's extra file reading from 
    // streams. Use a wildcard '*' to find metadata.json regardless of the 
    // parent directory name.
    string command = "unzip -p " + string(model_path) 
        + " '*/extra/metadata.json' 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe)
        throw system_error(
            errno, system_category(), "popen() for unzip failed");

    array< char, 128 > buffer;
    string metadata_json;
    while (fgets(buffer.data(), buffer.size(), pipe))
        metadata_json += buffer.data();

    if (int exit_code = pclose(pipe); exit_code != 0)
        throw system_error(errno, system_category(), "unzip command failed");

    if (metadata_json.empty())
        throw invalid_argument( 
            "Failed to extract metadata.json from model file.");

    // Call the other load_model overload that takes memory buffers.
    return load_model(
        {model_data.data(), static_cast<uint32_t>(model_data.size())},
        {metadata_json.data(), static_cast<uint32_t>(metadata_json.size())},
        device );
}

pair< unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model(
    DataBuffer model_buffer, DataBuffer metadata_buffer, torch::Device device )
{
    return make_pair(
        load_model( model_buffer, device ),
        Hyperparameters( string( metadata_buffer.data, metadata_buffer.len ))
    );
}

unique_ptr< torch::jit::script::Module > load_model(
    DataBuffer model_buffer, torch::Device device )
{
    static std::istringstream model_data_stream;
    // when reading the model from a string stream there seems to be a problem
    // with the embedded metadata, so we provided as an extra parameter
    model_data_stream.str( string( model_buffer.data, model_buffer.len ));

    auto model = make_unique< torch::jit::script::Module >( torch::jit::load(
        model_data_stream, device ));
    model->eval();

    return model;
}

Hyperparameters::Hyperparameters( string const& metadata_json )
{
    boost::json::value metadata = boost::json::parse( metadata_json );
    if (!metadata.is_object() 
            || !metadata.as_object().contains("self_play_config"))
        throw std::invalid_argument("Model metadata is missing or incomplete.");
    const auto& sp_config = metadata.at("self_play_config").as_object();

    c_base = get_required_value<float>(sp_config, "c_base");
    c_init = get_required_value<float>(sp_config, "c_init");
    dirichlet_alpha = get_required_value<float>(sp_config, "dirichlet_alpha");
    dirichlet_epsilon = get_required_value<float>(
        sp_config, "dirichlet_epsilon");
    simulations = get_required_value<int32_t>(sp_config, "simulations");
    opening_moves = get_required_value<int32_t>(sp_config, "opening_moves");
    parallel_games = get_value_with_default<size_t>(
        sp_config, "parallel_games", 1);
    parallel_simulations = get_value_with_default<size_t>(
        sp_config, "parallel_simulations", 10);
    max_batch_size = get_value_with_default<size_t>(
        sp_config, "max_batch_size", 100);
    nodes_per_block = get_value_with_default< size_t >(
        sp_config, "nodes_per_block", 50*simulations );
}

} // namespace libtorch