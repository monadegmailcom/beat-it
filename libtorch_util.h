#include <torch/script.h> // Main LibTorch header for loading models
#include <torch/torch.h>

#include <iostream>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

namespace libtorch {

struct InferenceRequest 
{
    float const* state;
    float* policies;
    std::promise< float > promise;
};

// This class manages a dedicated thread for running batched model inference.
class InferenceManager 
{
public:
    InferenceManager( 
        std::string&& model_data, size_t state_size, size_t policies_size,
        size_t max_batch_size = 128,
        std::chrono::milliseconds batch_timeout = std::chrono::milliseconds( 5 ));

    ~InferenceManager(); 

    void update_model(std::string&& new_model_data);

    // This is called by worker threads to queue a request for inference.
    // predicted value is returned in the future,
    // memory for predicted policies is provided by the caller.
    std::future< float > queue_request( float const* state, float* policies ); 

    // A synchronous, single-item prediction for performance comparison.
    // This bypasses the queue and runs inference immediately on the calling thread.
    float predict_sync(float const* state, float* policies);
private:
    void inference_loop(); 

    const size_t max_batch_size;
    const std::chrono::milliseconds batch_timeout;

    size_t state_size;
    size_t policies_size;
    std::vector< torch::Tensor > batch_tensors;
    std::istringstream model_data_stream;
    torch::jit::script::Module module; // The loaded TorchScript model
    torch::Device device;
    std::queue< InferenceRequest > request_queue;
    std::mutex queue_mutex;
    std::mutex module_update_mutex;
    std::condition_variable cv;
    std::atomic< bool > stop_flag;
    std::vector<size_t> batch_sizes_log;
    std::future< void > inference_future;
};

} // namespace libtorch {