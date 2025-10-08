#pragma once

#include "statistics.h"

#include <boost/lockfree/queue.hpp>

#include <functional>
#include <array>
#include <thread>
#include <semaphore>

namespace inference {

template< size_t P >
using callback_t = 
    void (*)( void*, void*, std::array< float, P > const&, float );

// request for nn inference.
template< size_t G, size_t P >
struct Request 
{
    // report out <data> for <node> back to <caller> via <callback>.
    callback_t< P > callback;
    void* caller;
    void* node;

    // in:
    std::array< float, G > state;
    // out:
    std::array< float, P > policies;
    float nn_value;
};

struct TimeStats // NOSONAR
{
    TimeStats( Statistics& stats, size_t divisor ) 
    : stats( stats ), divisor( divisor ) {}

    ~TimeStats()
    {
        const auto duration =
            std::chrono::duration<float, std::micro>(
                std::chrono::steady_clock::now() - start
            ) / divisor;
        stats.update( static_cast<size_t>(duration.count()));
    }

    std::chrono::steady_clock::time_point start = 
        std::chrono::steady_clock::now();
    Statistics& stats;
    size_t divisor;
};

/* prefer bandwidth over latency with a 2-step pipeline: 
   1. queue requests waiting for nn evaluation. usually involves gpu processing.
      one dedicated worker for inference queue processing.
   2. queue evaluated request waiting for client callback calls. 
      usually involves backpropagation on the client side.
      one dedicated worker for notify queue processing.
   for best performance assume that client callback calls takes much less time 
   than nn evaluation.
*/
template< size_t G, size_t P >
class Service
{
public:
    using request_type = Request< G, P >;

    explicit Service( size_t max_batch_size ) 
    : max_batch_size( max_batch_size ), inference_queue( max_batch_size ), 
      notify_queue( max_batch_size ), 
      available_inference_slots( max_batch_size ),
      available_notify_slots( max_batch_size )
    {
        if (!max_batch_size)
            throw std::source_location::current();
        inference_worker = std::jthread( &Service::run, this );
        notify_worker = std::jthread( &Service::notify_clients, this );
    }

    virtual ~Service()
    {
        // Signal the workers to stop.
        stop = true;
        // wake up threads waiting for nn evaluation.
        inference_queue.push( request_type());
        available_requests.release();
        // wake up threads waiting for client callback.
        notify_queue.push( request_type());
        available_notifications.release();
        
        // join workers.
        inference_worker.join();
        notify_worker.join();
    }

    // blocks if max queue size is reached. this slows down production of new
    // requests intentionally. 
    void push( request_type const& request )
    {
        // block if queue is full.
        available_inference_slots.acquire();

        while (!inference_queue.push( request ))
            std::this_thread::yield();

        available_requests.release();
    }

    Statistics const& batch_size_stats() const noexcept
    { return batch_size_stats_; }

    Statistics const& inference_time_stats() const noexcept
    { return inference_time_stats_; }

    void reset_stats() noexcept
    {
        batch_size_stats_.reset();
        inference_time_stats_.reset();
    }
protected:
    // promise: feed all batched requests into the nn and set nn_value and 
    // policies. 
    virtual void inference( request_type[], size_t batch_size ) = 0; 
private:
    void notify_clients()
    {
        request_type request;
        while (true)
        {
            available_notifications.acquire();
            if (stop)
                break;
            while (!notify_queue.pop( request ))
                std::this_thread::yield();

            available_notify_slots.release();
            request.callback( 
                request.caller, request.node, request.policies, 
                request.nn_value );
        }
    }

    void run()
    {
        std::vector< request_type > request_batch( max_batch_size ); 

        while (true)
        {
            // pop all available requests from the queue, but at most 
            // max_batch_size.
            size_t batch_size = 0;
            for (; !stop && batch_size != max_batch_size; ++batch_size)
                if (available_requests.try_acquire())
                    while (!inference_queue.pop( // NOSONAR
                                request_batch[batch_size] ))
                        std::this_thread::yield();
                else if (!batch_size)                    
                    // blocking wait on empty request queue.
                    available_requests.acquire(); 
                else 
                    break;
           
            if (stop)
                break;
            // empty slots in inference queue.
            available_inference_slots.release( batch_size );
            
            // process requests
            // call inference implementation and measure duration.
            { 
                TimeStats duration_per_request( 
                    inference_time_stats_, batch_size );
                inference( request_batch.data(), batch_size ); 
            }
          
            // push all batched requests into notify queue.
            for (size_t i = 0; i < batch_size; ++i)
            {
                // block if queue is full.
                available_notify_slots.acquire();

                while (!notify_queue.push( request_batch[i])) // NOSONAR
                    std::this_thread::yield(); 
            }

            available_notifications.release( batch_size );
            batch_size_stats_.update( batch_size );
        }
    }

    size_t max_batch_size;
    std::jthread inference_worker;
    std::jthread notify_worker;
    bool stop = false;
    boost::lockfree::queue< request_type > inference_queue;
    boost::lockfree::queue< request_type > notify_queue;
    Statistics batch_size_stats_;
    Statistics inference_time_stats_;
    std::counting_semaphore<> available_inference_slots;
    std::counting_semaphore<> available_notify_slots;
    std::counting_semaphore<> available_requests {0};
    std::counting_semaphore<> available_notifications {0};
};

} // inference