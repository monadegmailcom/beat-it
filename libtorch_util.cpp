#include "libtorch_util.h"

#include <map>
#include <iomanip>
#include <string>
#include <boost/json.hpp>

using namespace std;

namespace libtorch {

InferenceManager::InferenceManager( 
    std::string&& model_data , size_t state_size, size_t policies_size,
    size_t max_batch_size, std::chrono::milliseconds batch_timeout )
: max_batch_size( max_batch_size ), batch_timeout( batch_timeout ), state_size( state_size ), 
  policies_size( policies_size ), model_data_stream( std::move(model_data) ), 
  device( torch::kCPU), stop_flag( false )
{
    torch::jit::ExtraFilesMap extra_files;
    module = torch::jit::load(model_data_stream, extra_files);

    // Check if metadata exists and parse it using Boost.JSON
    if (extra_files.count("metadata.json")) 
    {
        try 
        {
            metadata = boost::json::parse(extra_files.at("metadata.json"));
            std::cout << "Successfully loaded and parsed model metadata with Boost.JSON." << std::endl;
        } 
        catch (const boost::system::system_error& e) {
            std::cerr << "Warning: Could not parse model metadata.json with Boost.JSON: " << e.what() << std::endl;
            // Initialize with an empty object to avoid errors on access
            metadata = boost::json::object();
        }
    } 
    else 
    {
        std::cout << "Warning: No metadata.json found in the model file." << std::endl;
        metadata = boost::json::object();
    }

    module.eval();
    
    // Check for available hardware backends, preferring MPS on Apple Silicon, then CUDA.
#if defined(__APPLE__) && defined(__aarch64__)
    if (torch::mps::is_available()) 
    {
        std::cout << "MPS is available! Moving model to Apple Silicon GPU." << std::endl;
        device = torch::kMPS;
    } 
    else
#endif
    if (torch::cuda::is_available()) 
    {
        std::cout << "CUDA is available! Moving model to NVIDIA GPU." << std::endl;
        device = torch::kCUDA;
    } 
    else 
    {
        std::cout << "No GPU backend found. Using CPU." << std::endl;
        device = torch::kCPU;
    }

    module.to(device);

    // Start the inference loop thread after everything is initialized.
    inference_future = async( &InferenceManager::inference_loop, this );
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

float InferenceManager::predict_sync(float const* state, float* policies)
{
    // This function is designed to be called directly from a worker thread.
    // It bypasses the queueing mechanism for direct, synchronous inference.
    // NOTE: This assumes that module.forward() is thread-safe for inference,
    // which is generally true for PyTorch models in eval mode.

    // 1. Create a 2D tensor of shape [1, state_size] from the raw pointer.
    auto input_tensor = torch::from_blob(
        const_cast<float*>(state), {1, (long)state_size}, torch::kFloat32);

    // 2. Move tensor to the correct device.
    input_tensor = input_tensor.to(device);

    // 3. Run inference.
    torch::jit::IValue output_ivalue = module.forward({input_tensor});
    auto output_tuple = output_ivalue.toTuple();

    // 4. Get results and move them to CPU.
    // The output tensors will have a batch dimension of 1.
    torch::Tensor value_tensor = output_tuple->elements()[0].toTensor().to(torch::kCPU);
    torch::Tensor policy_tensor = output_tuple->elements()[1].toTensor().to(torch::kCPU);

    // 5. Copy policy data to the output buffer.
    float* const policy_ptr = policy_tensor.data_ptr<float>();
    std::copy(policy_ptr, policy_ptr + policies_size, policies);

    // 6. Return the scalar value.
    return value_tensor[0].item<float>();
}

void InferenceManager::update_model(std::string&& new_model_data)
{
    // Load the new model from the byte stream into a temporary object.
    std::istringstream new_model_stream(std::move(new_model_data));
    torch::jit::ExtraFilesMap extra_files;
    auto new_module = torch::jit::load(new_model_stream, extra_files);
    new_module.to(device);
    new_module.eval();

    boost::json::value new_metadata;
    if (extra_files.count("metadata.json")) 
    {
        try 
        {
            new_metadata = boost::json::parse(extra_files.at("metadata.json"));
        } 
        catch (const boost::system::system_error& e) 
        {
            std::cerr << "Warning: Could not parse metadata in updated model: " << e.what() << std::endl;
            new_metadata = boost::json::object();
        }
    }

    // Lock and swap the new model into place. This is much more efficient
    // than destroying and recreating the entire InferenceManager.
    {
        lock_guard< mutex > lock( module_update_mutex);
        module = new_module;
        metadata = std::move(new_metadata);
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
            lock_guard< mutex > lock( module_update_mutex);
            output_ivalue = module.forward({input_batch});
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
