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

torch::Device get_device();

struct InferenceRequest
{
    float const* state;
    float* policies;
    std::promise< float > promise;
};

struct Hyperparameters
{
    // require: metadata_json has to contain a self_play_config sub-object
    //          with the variables below, otherwise it will throw
    // promise: json is parsed and assigned to the variables below
    Hyperparameters( std::string const& metadata_json );
    Hyperparameters() {};

    float c_base = 0.0f;
    float c_init = 0.0f;
    float dirichlet_alpha = 0.0f;
    float dirichlet_epsilon = 0.0f;
    size_t simulations = 0;
    size_t opening_moves = 0;
    size_t threads = 0;
};

// promise: - model is set to eval mode
//          - model is moved to the most powerful device on the machine
std::pair< std::unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model(
    const char* model_path );
std::pair< std::unique_ptr< torch::jit::script::Module >, Hyperparameters > load_model(
    char const* model_data, size_t model_data_len,
    const char* metadata_json, size_t metadata_len );

float sync_predict(
    torch::jit::script::Module& model,
    float const* game_state_players, size_t game_state_players_size,
    float* policies, size_t policies_size );

// This class manages a dedicated thread for running batched model inference.
class InferenceManager
{
public:
    InferenceManager(
        std::unique_ptr< torch::jit::script::Module >&&,
        const Hyperparameters& hp,
        size_t state_size, size_t policies_size,
        size_t max_batch_size = 128,
        std::chrono::milliseconds batch_timeout = std::chrono::milliseconds( 5 ));

    // be sure not to copy or assign the inference manager accidentally
    InferenceManager( const InferenceManager& ) = delete;
    InferenceManager& operator=( const InferenceManager& ) = delete;
    InferenceManager( InferenceManager&& ) = delete;
    InferenceManager& operator=( InferenceManager&&) = delete;

    ~InferenceManager();

    // threadsafe replacement of model
    void update_model( std::unique_ptr< torch::jit::script::Module >&&, const Hyperparameters& hp );

    // This is called by worker threads to queue a request for inference.
    // predicted value is returned in the future,
    // memory for predicted policies is provided by the caller.
    std::future< float > queue_request( float const* state, float* policies );

    // This moves the log data out of the manager. Should only be called once at the end.
    std::vector<size_t> const& get_inference_histogram() const;
private:
    void inference_loop();

    const size_t max_batch_size;
    const std::chrono::milliseconds batch_timeout;

    size_t state_size;
    size_t policies_size;
    std::vector< torch::Tensor > batch_tensors;
    torch::Device device;
    std::unique_ptr< torch::jit::script::Module > model;
    std::mutex model_update_mutex;
    std::queue< InferenceRequest > request_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic< bool > stop_flag;
    std::vector<size_t> inference_histogram;
    std::future< void > inference_future;
};

} // namespace libtorch {