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
      model( std::move( model )) {}

    // threadsafe replacement of model
    void update_model( 
        std::unique_ptr< torch::jit::script::Module >&& new_model )
    {
        std::scoped_lock _( model_update_mutex );
        model = std::move( new_model );
    }
private:
    void inference( 
        service_type::request_type request_batch[], size_t batch_size ) override
    {
        batch_tensors.clear();
        for (size_t i = 0; i < batch_size; ++i)
            batch_tensors.push_back( torch::from_blob(
                const_cast< float* >( request_batch[i].state.data()), // NOSONAR
                G, torch::kFloat32)); 
        torch::Tensor input_batch = torch::stack( batch_tensors).to( device);

        torch::jit::IValue output_ivalue;
        {
            // Lock the module while running inference to prevent it from being
            // replaced by an update call from another thread mid-operation.
            std::scoped_lock _( model_update_mutex);
            output_ivalue = model->forward({input_batch});
        }

        auto output_tuple = output_ivalue.toTuple();
        torch::Tensor value_batch = 
            output_tuple->elements()[0].toTensor().to( torch::kCPU);
        torch::Tensor policy_batch = 
            output_tuple->elements()[1].toTensor().to( torch::kCPU);
    
        for (size_t i = 0; i < batch_size; ++i)
        {
            auto& request = request_batch[i];
            inference::Response< P > response {
                .node = request.node,
                .nn_value = value_batch[i].item< float >()
            };
            std::copy_n(
                policy_batch[i].data_ptr< float >(), P,
                response.policies.begin());
            while (!request.response_queue->push( response ))
                std::this_thread::yield();
        }
    }

    std::vector< torch::Tensor > batch_tensors;
    torch::Device device;
    std::unique_ptr< torch::jit::script::Module > model;
    std::mutex model_update_mutex;
};

} // namespace libtorch