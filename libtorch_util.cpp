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
    // Check for available hardware backends, preferring MPS on Apple Silicon, then CUDA.
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
        throw runtime_error("Required key not found in self_play_config: '" + key + "'");
    return boost::json::value_to<T>(obj.at(key));
}

pair< unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model(
    const char* model_path, torch::Device device )
{
    // Read the entire model file into a string buffer.
    ifstream model_file(model_path, ios::binary);
    if (!model_file)
        throw runtime_error("Failed to open model file: " + string(model_path));

    stringstream model_buffer;
    model_buffer << model_file.rdbuf();
    string model_data = model_buffer.str();

    // Use `unzip` via `popen` to robustly extract the metadata.json.
    // This is more reliable than relying on libtorch's extra file reading from streams.
    // Use a wildcard '*' to find metadata.json regardless of the parent directory name
    // (e.g., 'final_model/extra/metadata.json' or 'checkpoint/extra/metadata.json').
    string command = "unzip -p " + string(model_path) + " '*/extra/metadata.json' 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe)
        throw runtime_error("popen() failed!");

    char buffer[128];
    string metadata_json;
    while (fgets(buffer, sizeof(buffer), pipe))
        metadata_json += buffer;

    if (int exit_code = pclose(pipe); exit_code != 0)
        throw runtime_error("unzip command failed with exit code " + to_string(exit_code));

    if (metadata_json.empty())
        throw runtime_error("Failed to extract metadata.json from model file. Is the path correct?");

    // Call the other load_model overload that takes memory buffers.
    return load_model(
        model_data.data(), model_data.size(),
        metadata_json.data(), metadata_json.size(),
        device );
}

pair< unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model(
    char const* model_data, size_t model_data_len,
    const char* metadata_json, size_t metadata_len, torch::Device device )
{
    static std::istringstream model_data_stream;
    // when reading the model from a string stream there seems to be a problem
    // with the embedded metadata, so we provided as an extra parameter
    model_data_stream.str( string( model_data, model_data_len ));

    auto model = make_unique< torch::jit::script::Module >( torch::jit::load(
        model_data_stream ));
    model->eval();

    auto hyperparameters( string(metadata_json, metadata_len));

    return make_pair( std::move( model ), hyperparameters );
}

Hyperparameters::Hyperparameters( string const& metadata_json )
{
    boost::json::value metadata = boost::json::parse( metadata_json );
    if (!metadata.is_object() || !metadata.as_object().contains("self_play_config"))
        throw std::runtime_error("Model metadata is missing or incomplete.");
    const auto& sp_config = metadata.at("self_play_config").as_object();

    c_base = get_required_value<float>(sp_config, "c_base");
    c_init = get_required_value<float>(sp_config, "c_init");
    dirichlet_alpha = get_required_value<float>(sp_config, "dirichlet_alpha");
    dirichlet_epsilon = get_required_value<float>(sp_config, "dirichlet_epsilon");
    simulations = get_required_value<int32_t>(sp_config, "simulations");
    opening_moves = get_required_value<int32_t>(sp_config, "opening_moves");
    threads = get_required_value<size_t>(sp_config, "threads");
}

float sync_predict(
    torch::jit::script::Module& model, torch::Device device,
    float const* game_state_players, size_t game_state_players_size,
    float* policies, size_t policies_size )
{
    auto input_tensor = torch::from_blob(
        const_cast< float* >( game_state_players ),
        {1, (long)game_state_players_size }, torch::kFloat32);

    // Move tensor to the correct device.
    input_tensor = input_tensor.to( device );

    // Run inference.
    torch::jit::IValue output_ivalue = model.forward( {input_tensor} );
    auto output_tuple = output_ivalue.toTuple();

    // Get results and move them to CPU.
    // The output tensors will have a batch dimension of 1.
    torch::Tensor value_tensor = output_tuple->elements()[0].toTensor().to( torch::kCPU );
    torch::Tensor policy_tensor = output_tuple->elements()[1].toTensor().to( torch::kCPU );

    // Copy policy data to the output buffer.
    float const* const policy_ptr = policy_tensor.data_ptr< float >();
    copy( policy_ptr, policy_ptr + policies_size, policies );

    return value_tensor[0].item< float >();
}

InferenceManager::InferenceManager(
    unique_ptr< torch::jit::script::Module >&& model,
    torch::Device device,
    Hyperparameters const& hp,
    size_t state_size, size_t policies_size,
    size_t max_batch_size, std::chrono::milliseconds batch_timeout )
: max_batch_size( max_batch_size ), batch_timeout( batch_timeout ), state_size( state_size ),
  policies_size( policies_size ), device( device ), model( std::move( model )), stop_flag( false ),
  inference_histogram( hp.threads + 1, 0 )
{
    // Start the inference loop thread after everything is initialized.
    inference_future = std::async( &InferenceManager::inference_loop, this );
}

InferenceManager::~InferenceManager()
{
    stop_flag = true;
    cv.notify_one();
    if (inference_future.valid())
        inference_future.wait();
}

vector< size_t > const& InferenceManager::get_inference_histogram() const
{
    return inference_histogram;
}

future< float > InferenceManager::queue_request( float const* state, float* policies )
{
    promise< float > promise;
    auto future = promise.get_future();
    {
        lock_guard< mutex > lock(queue_mutex);
        request_queue.push( { state, policies, std::move( promise )});
    }
    cv.notify_one();
    return future;
}

void InferenceManager::update_model(
    unique_ptr< torch::jit::script::Module >&& new_model,
    Hyperparameters const& hp )
{
    // Lock and swap the new model into place. This is much more efficient
    // than destroying and recreating the entire InferenceManager.
    lock_guard< mutex > lock( model_update_mutex );
    model = std::move( new_model );
    inference_histogram.resize( hp.threads + 1, 0 );
}

void InferenceManager::inference_loop()
{
    vector< InferenceRequest > batch;
    while (!stop_flag)
    {
        {
            unique_lock< mutex > lock( queue_mutex );
            // Wait until the queue has items or a timeout occurs.
            // The timeout is crucial to process incomplete batches with low latency.
            // lock is released while waiting
            cv.wait_for(
                lock, batch_timeout, [this] () { return !request_queue.empty() || stop_flag; });

            if (stop_flag)
                return;

            // Pull requests from the queue to form a batch.
            batch.clear();
            while (!request_queue.empty() && batch.size() < max_batch_size)
            {
                batch.push_back( std::move( request_queue.front()));
                request_queue.pop();
            }
        }

        if (batch.empty())
            continue;

        // Increment the histogram bin corresponding to the current batch size.
        const size_t batch_size = batch.size();
        if (batch_size < inference_histogram.size())
            inference_histogram[batch_size]++;

        // Prepare the batch tensor for the model.
        batch_tensors.clear();
        batch_tensors.reserve( batch.size());
        for (auto& req : batch)
            batch_tensors.push_back( torch::from_blob(
                const_cast< float* >( req.state ), state_size, torch::kFloat32));
        torch::Tensor input_batch = torch::stack( batch_tensors).to( device);

        torch::jit::IValue output_ivalue;
        {
            // Lock the module while running inference to prevent it from being
            // swapped out by an update call from another thread mid-operation.
            lock_guard< mutex > lock( model_update_mutex);
            output_ivalue = model->forward({input_batch});
        }

        auto output_tuple = output_ivalue.toTuple();
        torch::Tensor value_batch = output_tuple->elements()[0].toTensor().to( torch::kCPU);
        torch::Tensor policy_batch = output_tuple->elements()[1].toTensor().to( torch::kCPU);

        // Distribute the results back to the waiting threads.
        for (size_t i = 0; i < batch.size(); ++i)
        {
            // copy the policy items into the provided buffer from the request
            float* const policy_ptr = policy_batch[i].data_ptr< float >();
            copy( policy_ptr, policy_ptr + policies_size, batch[i].policies );

            batch[i].promise.set_value( value_batch[i].item< float >());
        }
    }
}

} // namespace libtorch {
