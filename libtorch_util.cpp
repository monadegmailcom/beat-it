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
        {model_data.data(), static_cast<int32_t>(model_data.size())},
        {metadata_json.data(), static_cast<int32_t>(metadata_json.size())},
        device );
}

pair< unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model(
    DataBuffer model_buffer,
    DataBuffer metadata_buffer, torch::Device device )
{
    static std::istringstream model_data_stream;
    // when reading the model from a string stream there seems to be a problem
    // with the embedded metadata, so we provided as an extra parameter
    model_data_stream.str( string( model_buffer.data, model_buffer.len ));

    auto model = make_unique< torch::jit::script::Module >( torch::jit::load(
        model_data_stream, device ));
    model->eval();

    return make_pair(
        std::move( model ),
        Hyperparameters( string( metadata_buffer.data, metadata_buffer.len ))
    );
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
    threads = get_required_value<size_t>(sp_config, "threads");
    selfplay_threads = get_value_with_default<size_t>(
        sp_config, "selfplay_threads", 10);
    max_batch_size = get_value_with_default<size_t>(
        sp_config, "max_batch_size", 100);
}

float sync_predict(
    torch::jit::script::Module& model, torch::Device device,
    float const* game_state_players, size_t game_state_players_size,
    float* policies, size_t policies_size )
{
    auto input_tensor = torch::from_blob(
        const_cast< float* >( game_state_players ), // NOSONAR
        {1, (long)game_state_players_size }, torch::kFloat32);

    // Move tensor to the correct device.
    input_tensor = input_tensor.to( device );

    // Run inference.
    torch::jit::IValue output_ivalue = model.forward( {input_tensor} );
    auto output_tuple = output_ivalue.toTuple();

    // Get results and move them to CPU.
    // The output tensors will have a batch dimension of 1.
    torch::Tensor value_tensor = output_tuple->elements()[0].toTensor().to( 
        torch::kCPU );
    torch::Tensor policy_tensor = output_tuple->elements()[1].toTensor().to( 
        torch::kCPU );

    // Copy policy data to the output buffer.
    float const* const policy_ptr = policy_tensor.data_ptr< float >();
    copy( policy_ptr, policy_ptr + policies_size, policies );

    return value_tensor[0].item< float >();
}

InferenceManager::InferenceManager(
    unique_ptr< torch::jit::script::Module >&& model,
    torch::Device device, size_t state_size, size_t policies_size,
    size_t max_queue_size )
:   state_size( state_size ), policies_size( policies_size ), device( device ), 
    model( std::move( model )), stop_flag( false ), 
    max_queue_size( max_queue_size )
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

future< float > InferenceManager::queue_request( 
    float const* state, float* policies )
{
    promise< float > promise;
    auto future = promise.get_future();
    {
        unique_lock lock(queue_mutex);
        // block if queue is full.
        if (request_queue.size() >= max_queue_size)
            queue_full_cv.wait(
                lock, 
                [this] { return request_queue.size() < max_queue_size; });
        request_queue.emplace( state, policies, std::move( promise ));
    }

    cv.notify_one();
    return future;
}

void InferenceManager::update_model(
    unique_ptr< torch::jit::script::Module >&& new_model )
{
    // Lock and swap the new model into place. This is much more efficient
    // than destroying and recreating the entire InferenceManager.
    scoped_lock< mutex > _( model_update_mutex );
    model = std::move( new_model );
}

void InferenceManager::reset_stats() noexcept
{
    queue_size_stats_.reset();
    inference_time_stats_.reset();
}

void InferenceManager::inference_loop()
{
    vector< InferenceRequest > request_batch;

    while (!stop_flag)
    {
        std::queue<InferenceRequest> local_queue;
        {
            unique_lock lock( queue_mutex );
            cv.wait(
                lock, [this] { return !request_queue.empty() || stop_flag; });

            if (stop_flag)
                break;

            // Swap with the member queue inside the lock. This is an O(1)
            // operation and minimizes the time the mutex is held.
            request_queue.swap(local_queue);
        }
        
        // notify potentially blocked threads waiting to fill the queue again.
        queue_full_cv.notify_all();

        request_batch.clear();
        request_batch.reserve(local_queue.size());
        while (!local_queue.empty()) {
            request_batch.push_back(std::move(local_queue.front()));
            local_queue.pop();
        }

        auto start = std::chrono::steady_clock::now();

        // Increment the histogram bin corresponding to the current batch size.
        queue_size_stats_.update( request_batch.size());

        // Prepare the batch tensor for the model.
        batch_tensors.clear();
        batch_tensors.reserve( request_batch.size());
        for (auto const& req : request_batch)
            batch_tensors.push_back( torch::from_blob(
                const_cast< float* >( req.state ), // NOSONAR
                state_size, torch::kFloat32)); 
        torch::Tensor input_batch = torch::stack( batch_tensors).to( device);

        torch::jit::IValue output_ivalue;
        {
            // Lock the module while running inference to prevent it from being
            // replaced by an update call from another thread mid-operation.
            scoped_lock< mutex > _( model_update_mutex);
            output_ivalue = model->forward({input_batch});
        }

        auto output_tuple = output_ivalue.toTuple();
        torch::Tensor value_batch = output_tuple->elements()[0].toTensor().to( torch::kCPU);
        torch::Tensor policy_batch = output_tuple->elements()[1].toTensor().to( torch::kCPU);

        // Distribute the results back to the waiting threads.
        for (size_t i = 0; i < request_batch.size(); ++i)
        {
            // copy the policy items into the provided buffer from the request
            float* const policy_ptr = policy_batch[i].data_ptr< float >();
            copy( policy_ptr, policy_ptr + policies_size, request_batch[i].policies );

            request_batch[i].promise.set_value( value_batch[i].item< float >());
        }

        const auto duration_per_item =
            std::chrono::duration<float, std::micro>(
                std::chrono::steady_clock::now() - start
            ) / request_batch.size();
        inference_time_stats_.update(
            static_cast<size_t>(duration_per_item.count()));
    }
}

} // namespace libtorch