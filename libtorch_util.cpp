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

pair< unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model( 
    const char* model_path )
{
    torch::jit::ExtraFilesMap extra_files;
    auto model = make_unique< torch::jit::script::Module >( torch::jit::load( 
        model_path, get_device(), extra_files ));
    model->eval();
    auto hyperparameters( extra_files.at( "metadata.json" ));

    return make_pair( std::move( model ), hyperparameters );
}

pair< unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model( 
    char const* model_data, size_t model_data_len, 
    const char* metadata_json, size_t metadata_len )
{
    static std::istringstream model_data_stream;
    // when reading the model from a string stream there seems to be a problem
    // with the embedded metadata, so we provided as an extra parameter
    model_data_stream.str( string( model_data, model_data_len ));
    
    auto model = make_unique< torch::jit::script::Module >( torch::jit::load( 
        model_data_stream, get_device()));
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
    
    c_base = boost::json::value_to<float>(sp_config.at("c_base"));
    c_init = boost::json::value_to<float>(sp_config.at("c_init"));
    dirichlet_alpha = boost::json::value_to<float>(sp_config.at("dirichlet_alpha"));
    dirichlet_epsilon = boost::json::value_to<float>(sp_config.at("dirichlet_epsilon"));
    simulations = boost::json::value_to<int32_t>(sp_config.at("simulations"));
    opening_moves = boost::json::value_to<int32_t>(sp_config.at("opening_moves"));
    threads = boost::json::value_to<size_t>(sp_config.at("threads"));
}

InferenceManager::InferenceManager( 
    unique_ptr< torch::jit::script::Module >&& model,
    const Hyperparameters& hp,
    size_t state_size, size_t policies_size,
    size_t max_batch_size, std::chrono::milliseconds batch_timeout )
: max_batch_size( max_batch_size ), batch_timeout( batch_timeout ), state_size( state_size ), 
  policies_size( policies_size ), device( get_device()), model( std::move( model )), stop_flag( false ),
  inference_histogram( hp.threads + 1, 0 )
{
    // Start the inference loop thread after everything is initialized.
    inference_future = async( &InferenceManager::inference_loop, this );
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
