#pragma once

#include "statistics.h"
#include "exception.h"

#include <boost/lockfree/queue.hpp>

#include <array>
#include <thread>
#include <semaphore>

namespace inference {

// response of nn inference.
template< size_t P >
struct Response
{
    void* node;
    std::array< float, P > policies;
    float nn_value;
};

// request for nn inference.
template< size_t G, size_t P >
struct Request 
{
    boost::lockfree::queue< Response< P >>* response_queue;
    void* node;
    std::array< float, G > state;
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
    using response_type = Response< P >;

    explicit Service( size_t max_batch_size )
    :   max_batch_size( max_batch_size ), inference_queue( max_batch_size ), 
        available_inference_slots( max_batch_size ) 
    {
        if (!max_batch_size)
            throw beat_it::Exception( "Max batch size cannot be zero.");
        inference_worker = std::jthread( &Service::run, this );
    }

    virtual ~Service() noexcept
    {
        // Signal the workers to stop.
        stop_source.request_stop();
        // poison pill to wake up threads waiting for nn evaluation.
        inference_queue.push( request_type());
        available_requests.release();
    }

    // blocks if max queue size is reached. this slows down production of new
    // requests intentionally. 
    // thread-safe.
    void push( request_type const& request )
    {
        while (!inference_queue.push( request ))
            // block if queue is full.
            available_inference_slots.acquire();

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

    size_t get_max_batch_size() const noexcept{ return max_batch_size; }
protected:
    // promise: feed all batched requests into the nn and push results into
    // response queues. 
    virtual void inference( request_type[], size_t batch_size ) = 0; 
private:
    // not thread-safe.
    void run()
    {
        std::vector< request_type > request_batch( max_batch_size ); 
        size_t batch_size = 0;

        while (!stop_source.stop_requested())
        {
            // pop all available requests from the queue, but at most 
            // max_batch_size.
            if (   batch_size != max_batch_size 
                && inference_queue.pop( request_batch[batch_size] ))
            {
                // signal available slots in inference queue. 
                available_inference_slots.release();
                ++batch_size;
            }
            else if (batch_size)
            {
                // process requests
                // call inference implementation and measure duration.
                { 
                    TimeStats duration_per_request( 
                        inference_time_stats_, batch_size );
                    inference( request_batch.data(), batch_size ); 
                }
                batch_size_stats_.update( batch_size );
                batch_size = 0;
            }
            else
                // blocking wait on empty request queue.
                available_requests.acquire(); 
        }
    }

    size_t max_batch_size;
    std::jthread inference_worker;
    std::stop_source stop_source; 
    boost::lockfree::queue< request_type > inference_queue;
    Statistics batch_size_stats_;
    Statistics inference_time_stats_;
    std::counting_semaphore<> available_inference_slots;
    std::counting_semaphore<> available_requests {0};
};

} // inference