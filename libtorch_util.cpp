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

torch::Device check_device()
{
    // Check for available hardware backends, preferring MPS on Apple Silicon, then CUDA.
    if (torch::mps::is_available()) 
    {
        cout << "MPS is available! Moving model to Apple Silicon GPU." << endl;
        return torch::kMPS;
    } 
    else if (torch::cuda::is_available()) 
    {
        cout << "CUDA is available! Moving model to NVIDIA GPU." << endl;
        return torch::kCUDA;
    } 
    else 
    {
        cout << "No GPU backend found. Using CPU." << endl;
        return torch::kCPU;
    }
}

InferenceManager::InferenceManager( 
    string&& model_data, torch::Device device, 
    size_t state_size, size_t policies_size,
    size_t max_batch_size, std::chrono::milliseconds batch_timeout )
: max_batch_size( max_batch_size ), batch_timeout( batch_timeout ), state_size( state_size ), 
  policies_size( policies_size ), model_data_stream( std::move(model_data)), 
  device( device ), stop_flag( false )
{
    torch::jit::ExtraFilesMap extra_files;
    model = torch::jit::load( model_data_stream, device, extra_files );
    metadata = boost::json::parse( extra_files.at( "metadata.json" ));

    initialize();
}

InferenceManager::InferenceManager(
    const char* model_path, torch::Device device,
    size_t state_size, size_t policies_size,
    size_t max_batch_size, std::chrono::milliseconds batch_timeout)
: max_batch_size( max_batch_size ), batch_timeout( batch_timeout ), state_size( state_size ), 
  policies_size( policies_size ), device( device ), stop_flag( false )
{
    torch::jit::ExtraFilesMap extra_files;
    model = torch::jit::load( model_path, device, extra_files );  
    metadata = boost::json::parse(extra_files.at("metadata.json"));

    initialize();
}

InferenceManager::~InferenceManager() 
{
    stop_flag = true;
    cv.notify_one();
    if (inference_future.valid())
        inference_future.wait();

    if (batch_sizes_log.empty()) {
        cout << "\n--- No inference batches were processed. ---" << endl;
        return;
    }

    map<size_t, size_t> counts;
    size_t max_count = 0;
    for (size_t size : batch_sizes_log) {
        counts[size]++;
        if (counts[size] > max_count) {
            max_count = counts[size];
        }
    }

    const size_t max_bar_width = 50;
    const double scale = (max_count > max_bar_width) ? static_cast<double>(max_bar_width) / max_count : 1.0;

    cout << "\n--- Inference Batch Size Histogram ---" << endl;
    cout << "Size | Freq.                                              | Count" << endl;
    cout << "--------------------------------------------------------------------" << endl;
    for (const auto& [size, count] : counts)
        cout << setw(4) << size << " | " << left << setw(max_bar_width) << string(static_cast<size_t>(count * scale), '*') << " | " << count << endl;
    cout << "--------------------------------------------------------------------" << endl;
}

void InferenceManager::initialize()
{
    model.eval();

    // Start the inference loop thread after everything is initialized.
    inference_future = async( &InferenceManager::inference_loop, this );
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

const boost::json::value& InferenceManager::get_metadata() const
{
    return metadata;
}

void InferenceManager::update_model( string&& new_model_data )
{
    torch::jit::ExtraFilesMap extra_files;

    model_data_stream.str( std::move( new_model_data ));
    auto new_model = torch::jit::load( model_data_stream, device, extra_files );
    boost::json::value new_metadata = boost::json::parse( extra_files.at( "metadata.json" ));
    
    // Lock and swap the new model into place. This is much more efficient
    // than destroying and recreating the entire InferenceManager.
    {
        lock_guard< mutex > lock( model_update_mutex );
        model = new_model;
        metadata = std::move( new_metadata );
    }
    std::cout << "InferenceManager model updated in-place." << std::endl;
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
                lock, 
                batch_timeout, 
                [this] () { return !request_queue.empty() || stop_flag; }
                );

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

        batch_sizes_log.push_back(batch.size());

        // Prepare the batch tensor for the model.
        batch_tensors.clear();
        batch_tensors.reserve( batch.size());
        for (auto& req : batch) 
            // cast to void* because of legacy c interface
            batch_tensors.push_back( torch::from_blob( 
                const_cast< float* >( req.state ), state_size, torch::kFloat32));
        torch::Tensor input_batch = torch::stack( batch_tensors).to( device);
        
        torch::jit::IValue output_ivalue;
        {
            // Lock the module while running inference to prevent it from being
            // swapped out by an update call from another thread mid-operation.
            lock_guard< mutex > lock( model_update_mutex);
            output_ivalue = model.forward({input_batch});
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
