#pragma once

#include "node.h"
#include "statistics.h"
#include "alphazero.h"

#include <torch/script.h> // Main LibTorch header for loading models
#include <torch/torch.h>

#include <boost/json.hpp>

#include <iostream>
#include <future>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

namespace libtorch {

struct DataBuffer {
    const char* data;
    int32_t len;
};

torch::Device get_device();

struct InferenceRequest
{
    // needed for usage in std::deque
    InferenceRequest(
        float const* state, float* policies, std::promise< float >&& promise )
    : state( state ), policies( policies ), promise( std::move( promise ) ) {}

    InferenceRequest() = delete;
    InferenceRequest( const InferenceRequest& ) = delete;
    InferenceRequest& operator=( const InferenceRequest& ) = delete;
    InferenceRequest( InferenceRequest&& ) noexcept = default;
    InferenceRequest& operator=( InferenceRequest&& ) noexcept = default;

    float const* state;
    float* policies;
    std::promise< float > promise;
};

struct Hyperparameters
{
    Hyperparameters() = default;
    // require: metadata_json has to contain a self_play_config sub-object
    //          with the variables below, otherwise it will throw
    // promise: json is parsed and assigned to the variables below
    explicit Hyperparameters( std::string const& metadata_json );

    float c_base = 0.0f;
    float c_init = 0.0f;
    float dirichlet_alpha = 0.0f;
    float dirichlet_epsilon = 0.0f;
    size_t simulations = 0;
    size_t opening_moves = 0;
    size_t threads = 0;
    size_t selfplay_threads = 0;
    size_t min_batch_size = 0;
};

// promise: model is set to eval mode
std::pair< std::unique_ptr< torch::jit::script::Module >,
           Hyperparameters > load_model(
                const char* model_path, torch::Device );
// promise: model is set to eval mode
std::pair< std::unique_ptr< torch::jit::script::Module >,
           Hyperparameters > load_model(
    DataBuffer model_buffer,
    DataBuffer metadata_buffer, torch::Device );

float sync_predict(
    torch::jit::script::Module& model, torch::Device,
    float const* game_state_players, size_t game_state_players_size,
    float* policies, size_t policies_size );

// This class manages a dedicated thread for running batched model inference.
class InferenceManager
{
public:
    InferenceManager(
        std::unique_ptr< torch::jit::script::Module >&&, torch::Device,
        size_t threads, size_t state_size, size_t policies_size,
        size_t min_batch_size, size_t max_batch_size,
        std::chrono::milliseconds batch_timeout
            = std::chrono::milliseconds( 5 ));

    // be sure not to copy or assign the inference manager accidentally
    InferenceManager() = delete;
    InferenceManager( const InferenceManager& ) = delete;
    InferenceManager& operator=( const InferenceManager& ) = delete;
    InferenceManager( InferenceManager&& ) = delete;
    InferenceManager& operator=( InferenceManager&&) = delete;

    ~InferenceManager();

    // threadsafe replacement of model
    void update_model( std::unique_ptr< torch::jit::script::Module >&& );

    // This is called by worker threads to queue a request for inference.
    // predicted value is returned in the future,
    // memory for predicted policies is provided by the caller.
    std::future< float > queue_request( float const* state, float* policies );

    std::vector<size_t> const& get_inference_histogram() const;
    Statistics const& queue_size_stats() const noexcept
        { return queue_size_stats_; }
    Statistics const& inference_time_stats() const noexcept
        { return inference_time_stats_; }
    void set_min_batch_size( size_t size ) noexcept
    { min_batch_size = size; }
    void reset_stats() noexcept;
private:
    void inference_loop();

    size_t min_batch_size;
    size_t max_batch_size;
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
    Statistics queue_size_stats_;
    Statistics inference_time_stats_;
};

namespace async {

template< typename BasePlayerT >
class Player : public BasePlayerT
{
public:
    Player( typename BasePlayerT::game_type game,
            alphazero::params::Ucb const& ucb,
            alphazero::params::GamePlay const& game_play,
            unsigned seed,
            NodeAllocator< typename BasePlayerT::value_type >& allocator,
            InferenceManager& im )
        : BasePlayerT( std::move(game), ucb, game_play, seed, allocator ),
          inference_manager( im ) {}
protected:
    InferenceManager& inference_manager;

    std::pair< float, std::array< float, BasePlayerT::policy_size > > predict(
        std::array< float, BasePlayerT::game_size > const& game_state_players ) override
    {
        // provide the buffer to copy predicted policies into
        std::array< float, BasePlayerT::policy_size > policies;
        // blocking call
        float value = async_predict( inference_manager, game_state_players.data(),
            policies.data());
        return std::make_pair( value, policies );
    }
};

} // namespace async

float async_predict( InferenceManager&, float const* game_state_players, float* policies );

} // namespace libtorch