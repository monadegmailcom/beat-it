#pragma once

#include "inference.h"
#include "statistics.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <boost/json.hpp>

#include <mutex>
#include <torch/cuda.h>
#ifdef __APPLE__
#include <torch/mps.h>
#endif

namespace libtorch
{

struct DataBuffer
{
    const char* data;
    uint32_t len;
};

torch::Device get_device();
// global mutex for mps device synchronization
std::mutex& get_mps_mutex();

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
    size_t parallel_games = 0;
    size_t parallel_simulations = 0;
    size_t max_batch_size = 0;
    size_t nodes_per_block = 0;
};

struct MatchupPlayerConfig
{
    int32_t type; // 1: mcts, 2: minimax, 3: tree_minimax
    uint32_t simulations_or_depth;
    const char* model_data;
    uint32_t model_data_len;
    libtorch::Hyperparameters hp;
};

// promise: model is set to eval mode
std::pair< std::unique_ptr< torch::jit::script::Module >, Hyperparameters >
load_model( const char* model_path, torch::Device );
// promise: model is set to eval mode
std::pair< std::unique_ptr< torch::jit::script::Module >, Hyperparameters >
load_model( DataBuffer model_buffer, DataBuffer metadata_buffer,
            torch::Device );
// promise: model is set to eval mode
std::unique_ptr< torch::jit::script::Module >
load_model( DataBuffer model_buffer, torch::Device );

template < size_t G, size_t P >
class InferenceService : public inference::Service< G, P >
{
  public:
    using service_type = inference::Service< G, P >;
    InferenceService( DataBuffer model_buffer,
                      torch::Device device, size_t max_batch_size )
        : service_type( max_batch_size, device.type() == torch::kCUDA ? std::max<size_t>(1, torch::cuda::device_count()) : 1 )
    {
        size_t num_workers = device.type() == torch::kCUDA ? std::max<size_t>(1, torch::cuda::device_count()) : 1;
        
        auto cpu_options = torch::TensorOptions().dtype( torch::kFloat32 );
        if ( device.type() == torch::kCUDA )
        {
            cpu_options = cpu_options.pinned_memory( true );
        }
        
        for ( size_t i = 0; i < num_workers; ++i )
        {
            torch::Device worker_device = device;
            if ( device.type() == torch::kCUDA )
                worker_device = torch::Device(torch::kCUDA, i);

            torch::DeviceGuard device_guard(worker_device);
            auto worker = std::make_unique<WorkerState>(worker_device);
            worker->model = load_model(model_buffer, worker_device);

            worker->cpu_input_tensor = torch::empty(
                { static_cast< long >( max_batch_size ), static_cast< long >( G ) },
                cpu_options );

            auto gpu_options =
                torch::TensorOptions().device( worker_device ).dtype( torch::kFloat32 );
            worker->gpu_input_tensor = torch::empty(
                { static_cast< long >( max_batch_size ), static_cast< long >( G ) },
                gpu_options );

            worker->cpu_value_tensor = torch::empty(
                { static_cast< long >( max_batch_size ), 1 }, cpu_options );
            worker->cpu_policy_tensor = torch::empty(
                { static_cast< long >( max_batch_size ), static_cast< long >( P ) },
                cpu_options );
                
            workers.push_back(std::move(worker));
        }
    }

    ~InferenceService()
    {
        for ( auto& worker : workers )
        {
            if ( worker->device.type() == torch::kCUDA )
            {
                // Synchronize on the specific device
                torch::cuda::synchronize(worker->device.index());
            }
            else if ( worker->device.type() == torch::kMPS )
            {
#ifdef __APPLE__
                torch::mps::synchronize();
#endif
            }
        }
    }

    // threadsafe replacement of model
    void
    update_model( DataBuffer model_buffer,
                  Statistics& batch_size_stats,
                  Statistics& inference_time_stats )
    {
        for ( auto& worker : workers )
        {
            std::scoped_lock _( worker->model_update_mutex );
            torch::DeviceGuard device_guard(worker->device);
            worker->model = load_model(model_buffer, worker->device);
        }

        batch_size_stats = Statistics();
        inference_time_stats = Statistics();
        
        for ( auto& worker : workers )
        {
            std::scoped_lock _( worker->model_update_mutex );
            batch_size_stats.join( worker->batch_size_stats_ );
            inference_time_stats.join( worker->inference_time_stats_ );
            worker->batch_size_stats_.reset();
            worker->inference_time_stats_.reset();
        }
    }

    Statistics const& batch_size_stats() const noexcept
    {
        std::scoped_lock _( aggregated_stats_mutex );
        aggregated_batch_size_stats_ = Statistics();
        for ( auto& worker : workers )
        {
            std::scoped_lock _( worker->model_update_mutex );
            aggregated_batch_size_stats_.join( worker->batch_size_stats_ );
        }
        return aggregated_batch_size_stats_;
    }

    Statistics const& inference_time_stats() const noexcept
    {
        std::scoped_lock _( aggregated_stats_mutex );
        aggregated_inference_time_stats_ = Statistics();
        for ( auto& worker : workers )
        {
            std::scoped_lock _( worker->model_update_mutex );
            aggregated_inference_time_stats_.join( worker->inference_time_stats_ );
        }
        return aggregated_inference_time_stats_;
    }

    // not thread-safe.
    void reset_stats() noexcept
    {
        for ( auto& worker : workers )
        {
            worker->batch_size_stats_.reset();
            worker->inference_time_stats_.reset();
        }
    }

    void pause_inference()
    {
        inference_paused.store(true, std::memory_order_release);
    }

    void resume_inference()
    {
        inference_paused.store(false, std::memory_order_release);
    }

  private:
    void inference( size_t worker_id, service_type::request_type request_batch[],
                    service_type::response_type response_batch[],
                    size_t batch_size ) override
    {
        auto& worker = workers[worker_id];

        // Gracefully yield the GPU if Python requested a pause for PyTorch training
        while (inference_paused.load(std::memory_order_acquire))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Lock the module while running inference to prevent it from being
        // replaced by an update call from another thread mid-operation.
        std::scoped_lock _( worker->model_update_mutex );
        torch::DeviceGuard device_guard(worker->device);

        // serialize inference on mps device to prevent metal command buffer
        // commit errors.
        std::unique_lock< std::mutex > mps_lock;
        if ( worker->device.type() == torch::kMPS )
            mps_lock = std::unique_lock< std::mutex >( get_mps_mutex() );

        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        // copy data to cpu tensor.
        auto cpu_input_view = worker->cpu_input_tensor.narrow( 0, 0, batch_size );
        float* tensor_data_ptr = cpu_input_view.template data_ptr< float >();
        for ( size_t i = 0; i < batch_size; ++i )
        {
            std::copy_n( request_batch[i].state.data(), G,
                         tensor_data_ptr + i * G );
        }

        torch::Tensor cpu_value_view;
        torch::Tensor cpu_policy_view;

        {
            auto gpu_input_view = worker->gpu_input_tensor.narrow( 0, 0, batch_size );
            // copy data to gpu asynchronously.
            gpu_input_view.copy_( cpu_input_view, true );

            // set mode to no model training
            c10::InferenceMode guard;
            torch::jit::IValue output_ivalue =
                worker->model->forward( { gpu_input_view } );

            auto output_tuple = output_ivalue.toTuple();
            auto gpu_value_batch =
                output_tuple->elements()[0].toTensor();
            auto gpu_policy_batch =
                output_tuple->elements()[1].toTensor();

            cpu_value_view = worker->cpu_value_tensor.narrow( 0, 0, batch_size );
            cpu_policy_view = worker->cpu_policy_tensor.narrow( 0, 0, batch_size );
            cpu_value_view.copy_( gpu_value_batch, true );
            cpu_policy_view.copy_( gpu_policy_batch, true );
        }

        const auto duration = std::chrono::duration< float, std::micro >(
                                  std::chrono::steady_clock::now() - start ) /
                              batch_size;

        worker->inference_time_stats_.update(
            static_cast< size_t >( duration.count() ) );
        worker->batch_size_stats_.update( batch_size );

        // copy data from cpu tensor to response structures.
        for ( size_t i = 0; i < batch_size; ++i )
        {
            auto& request = request_batch[i];
            auto& response = response_batch[i];

            response.node = request.node;
            response.nn_value = cpu_value_view[i].template item< float >();

            std::copy_n( cpu_policy_view[i].template data_ptr< float >(), P,
                         response.policies.begin() );
        }
    }

    struct WorkerState {
        torch::Device device;
        std::unique_ptr<torch::jit::script::Module> model;
        torch::Tensor cpu_input_tensor;
        torch::Tensor gpu_input_tensor;
        torch::Tensor cpu_value_tensor;
        torch::Tensor cpu_policy_tensor;
        Statistics batch_size_stats_;
        Statistics inference_time_stats_;
        mutable std::mutex model_update_mutex;
        
        WorkerState(torch::Device d) : device(d) {}
    };
    
    std::vector<std::unique_ptr<WorkerState>> workers;
    mutable std::mutex aggregated_stats_mutex;
    mutable Statistics aggregated_batch_size_stats_;
    mutable Statistics aggregated_inference_time_stats_;
    std::atomic<bool> inference_paused{false};
};

} // namespace libtorch