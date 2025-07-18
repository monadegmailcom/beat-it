#include <torch/script.h> // Main LibTorch header for loading models
#include <torch/torch.h>

#include <iostream>
#include <future>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <boost/json.hpp>

namespace libtorch {

torch::Device check_device();

struct InferenceRequest 
{
    float const* state;
    float* policies;
    std::promise< float > promise;
};

struct Hyperparameters 
{
    float c_base;
    float c_init;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    size_t simulations;
    size_t opening_moves;
};


Hyperparameters parse_hyperparameters( const std::string& metadata_json );

// This class manages a dedicated thread for running batched model inference.
class InferenceManager 
{
public:
    // construct by file path
    InferenceManager(
        const char* model_path,
        torch::Device device,
        size_t state_size, size_t policies_size,
        size_t max_batch_size = 128,
        std::chrono::milliseconds batch_timeout = std::chrono::milliseconds( 5 ));

    InferenceManager(
        // construct by memory
        char const* model_data,
        size_t model_data_len,
        const char* metadata_json,
        size_t metadata_len,
        torch::Device device,
        size_t state_size, size_t policies_size,
        size_t max_batch_size = 128,
        std::chrono::milliseconds batch_timeout = std::chrono::milliseconds( 5 ));

    InferenceManager( const InferenceManager& ) = delete;
    InferenceManager& operator=( const InferenceManager& ) = delete;
    InferenceManager( InferenceManager&& ) = delete;
    InferenceManager& operator=( InferenceManager&&) = delete;

    ~InferenceManager(); 

    // threadsafe replace of model and metadata
    void update_model( char const* new_model_data, size_t new_model_data_len, 
                       const char* new_metadata_json, size_t new_metadata_len );

    // This is called by worker threads to queue a request for inference.
    // predicted value is returned in the future,
    // memory for predicted policies is provided by the caller.
    std::future< float > queue_request( float const* state, float* policies ); 

    // threadsafe hyper parameter copy
    Hyperparameters get_hyperparameters();

    // This moves the log data out of the manager. Should only be called once at the end.
    std::vector<size_t> get_batch_sizes_log();
private:
    void initialize();
    void inference_loop();

    const size_t max_batch_size;
    const std::chrono::milliseconds batch_timeout;

    size_t state_size;
    size_t policies_size;
    std::vector< torch::Tensor > batch_tensors;
    std::istringstream model_data_stream;
    torch::Device device;
    torch::jit::script::Module model; // The loaded TorchScript model
    std::queue< InferenceRequest > request_queue;
    std::mutex queue_mutex;
    std::shared_mutex model_update_mutex;
    std::condition_variable cv;
    std::atomic< bool > stop_flag;
    std::vector<size_t> batch_sizes_log;
    std::future< void > inference_future;
    Hyperparameters hyperparameters;
};

} // namespace libtorch {