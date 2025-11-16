#pragma once

#include "node.h"
#include "statistics.h"
#include "inference.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <boost/json.hpp>

#include <iostream>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

namespace libtorch {

struct DataBuffer {
    const char* data;
    uint32_t len;
};

torch::Device get_device();

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

// promise: model is set to eval mode
std::pair< std::unique_ptr< torch::jit::script::Module >,
           Hyperparameters > load_model(
                const char* model_path, torch::Device );
// promise: model is set to eval mode
std::pair< std::unique_ptr< torch::jit::script::Module >,
           Hyperparameters > load_model(
    DataBuffer model_buffer,
    DataBuffer metadata_buffer, torch::Device );
// promise: model is set to eval mode
std::unique_ptr< torch::jit::script::Module > load_model(
    DataBuffer model_buffer, torch::Device );

template< size_t G, size_t P >
class InferenceService : public inference::Service< G, P >
{
public:
    using service_type = inference::Service< G, P >;
    InferenceService(
        std::unique_ptr< torch::jit::script::Module >&& model, 
        torch::Device device, size_t max_batch_size ) 
    : service_type( max_batch_size ), device( device ), 
      model( std::move( model ))
    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        if (device.type() == torch::kCUDA) {
            options = options.pinned_memory(true);
        }
        cpu_input_tensor = torch::empty(
            { static_cast<long>(max_batch_size), static_cast<long>(G) }, 
            options);
        cpu_value_tensor = torch::empty(
            { static_cast<long>(max_batch_size), 1 },
            options);
        cpu_policy_tensor = torch::empty(
            { static_cast<long>(max_batch_size), static_cast<long>(P) },
            options);
    }

    // threadsafe replacement of model
    void update_model( 
        std::unique_ptr< torch::jit::script::Module >&& new_model )
    {
        std::scoped_lock _( model_update_mutex );
        model = std::move( new_model );
    }
private:
    void inference( 
        service_type::request_type request_batch[], 
        service_type::response_type response_batch[], 
        size_t batch_size ) override
    {
        auto cpu_input_view = cpu_input_tensor.narrow(0, 0, batch_size);
        float* tensor_data_ptr = cpu_input_view.template data_ptr<float>();
        for (size_t i = 0; i < batch_size; ++i) {
            std::copy_n(
                request_batch[i].state.data(),
                G,
                tensor_data_ptr + i * G
            );
        }
        
        torch::Tensor cpu_value_view; 
        torch::Tensor cpu_policy_view;

        {
            // Lock the module while running inference to prevent it from being
            // replaced by an update call from another thread mid-operation.
            std::scoped_lock _( model_update_mutex);

            auto gpu_input = cpu_input_view.to(device, true);
            // no model training, seems to have affect on the memory handling,
            // if not set, crashes with mps gpu 
            c10::InferenceMode guard;
            torch::jit::IValue output_ivalue = model->forward({gpu_input});

            auto output_tuple = output_ivalue.toTuple();
            
            auto gpu_value_batch = output_tuple->elements()[0].toTensor();
            auto gpu_policy_batch = output_tuple->elements()[1].toTensor();

            cpu_value_view = cpu_value_tensor.narrow(0, 0, batch_size);
            cpu_policy_view = cpu_policy_tensor.narrow(0, 0, batch_size);
            cpu_value_view.copy_(gpu_value_batch, true);
            cpu_policy_view.copy_(gpu_policy_batch, true);
            if (device.type() == torch::kMPS)
                torch::mps::synchronize();
            else if (device.type() == torch::kCUDA)
                torch::cuda::synchronize();
        }

        for (size_t i = 0; i < batch_size; ++i)
        {
            auto& request = request_batch[i];
            auto& response = response_batch[i];

            response.node = request.node;
            response.nn_value = cpu_value_view[i].template item< float >();
            
            std::copy_n(
                cpu_policy_view[i].template data_ptr< float >(), P,
                response.policies.begin());
        }
    }

    torch::Device device;
    std::unique_ptr< torch::jit::script::Module > model;
    torch::Tensor cpu_input_tensor;
    torch::Tensor cpu_value_tensor;
    torch::Tensor cpu_policy_tensor;
    std::mutex model_update_mutex;
};

} // namespace libtorch