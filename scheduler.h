#pragma once
#include <thread>
#include <vector>

#include "statistics.h"

struct SchedulerStats
{
    Statistics busy_threads;
    Statistics blocked_threads;
    Statistics pending_threads;
    Statistics idle_threads;
};

/*
Scheduler class to keep the cpu busy. it tries to execute a task concurrently 
using a constant number of busy threads. typically your task has at least one 
blocking operation. it creates new threads on demand and does not destroy them
until deletion. if there is nothing to do or the maximum number of busy threads
is reached these threads are idle and do not consume cpu usage. 
Usage:
- derive publicly from Scheduler and pass the maximum number of threads that
  should run in busy state (not blocking for a async operation).
- override
    - void task(): the task to be executed concurrently.
    - bool completed(): indicates completion.
- start tasks with Scheduler::run(). it executes tasks concurrently and
  blocks until completion. you can repeat Scheduler::run() calls multiple 
  times non-concurrently.
- call a blocking operation in a async scope:
  { auto _( Scheduler::async_section()); blocking_operation(); }
  the scheduler will start a new task execution while you are waiting for the
  blocking operation to return without exceeding the maximum busy thread count.
  on leaving the async scope it blocks until the maximum busy thread count is 
  not exceeded.
*/
class Scheduler
{
protected:
    struct Async
    {
        explicit Async( Scheduler& scheduler ) : scheduler( scheduler ) 
        {
            scheduler.enter_async_section();
        }

        ~Async()
        {
            scheduler.leave_async_section();
        }

        Async( Async const& ) = delete;
        Async& operator=( Async const& ) = delete;
        Async( Async&& ) = delete;
        Async& operator=( Async&& ) = delete;

        Scheduler& scheduler;
    };

    friend struct Async;

    explicit Scheduler( 
        size_t max_number_of_busy_threads, 
        size_t max_number_of_threads_totally,
        SchedulerStats& stats)
        : max_number_of_busy_threads( max_number_of_busy_threads),
          max_number_of_threads_totally( max_number_of_threads_totally),
          stats( stats )
    {
        if (!max_number_of_busy_threads)
            throw std::source_location::current();
        if (!max_number_of_threads_totally)
            throw std::source_location::current();
    }
    
    virtual ~Scheduler() 
    {
        stop = true;
        idle_threads_cv.notify_all();
        for (auto& thread : thread_pool)
            thread.join();
    }

    // blocks until there are no more tasks to schedule and all tasks finished.
    void run()
    {
        std::unique_lock lock( scheduler_mutex );

        activate_worker();

        // wait until all tasks are finished.
        scheduler_cv.wait( 
            lock, 
            [this]
            { 
                return completed()
                    && !number_of_busy_threads
                    && !number_of_blocked_threads
                    && !number_of_pending_threads; 
            });
    }

    // create if you are about to enter a blocking operation. It will
    // execute task() asynchronously to keep the cpu busy.
    Async async_section()
    {
        return Async( *this );
    }

    virtual void task() = 0;
    virtual bool completed() = 0;
private:    
    void enter_async_section()
    {
        std::unique_lock lock( scheduler_mutex );
        
        ++number_of_blocked_threads;
        stats.blocked_threads.update( number_of_blocked_threads );

        --number_of_busy_threads;
        stats.busy_threads.update( number_of_busy_threads );

        activate_worker();
    }
    
    void leave_async_section()
    {
        std::unique_lock lock( scheduler_mutex );

        --number_of_blocked_threads;
        stats.blocked_threads.update( number_of_blocked_threads );

        if (number_of_busy_threads < max_number_of_busy_threads)
        { // thread is allowed to proceed.
            ++number_of_busy_threads;
            stats.busy_threads.update( number_of_busy_threads );
        }
        else
        { // thread has to to enter pending state and wait.
            ++number_of_pending_threads;
            stats.pending_threads.update( number_of_pending_threads );

            // note: because the current thread was blocked the number of busy
            // threads does not decrease when it enters pending state.
          
            pending_threads_cv.wait(
                lock, 
                [this] 
                { 
                    return number_of_busy_threads < max_number_of_busy_threads; 
                });

            --number_of_pending_threads;
            stats.pending_threads.update( number_of_pending_threads );

            ++number_of_busy_threads;
            stats.busy_threads.update( number_of_busy_threads );
        }
    }

    void worker()
    {
        while (!stop)
        {
            // run task unlocked
            task(); 

            std::unique_lock lock( scheduler_mutex );

            // go to idle state if nothing more is to be done.
            if (completed())
            {
                --number_of_busy_threads;
                stats.busy_threads.update( number_of_busy_threads );

                ++number_of_idle_threads;
                stats.idle_threads.update( number_of_idle_threads );

                // notify scheduler so it can leave run method.
                scheduler_cv.notify_one();
                pending_threads_cv.notify_one();
                
                idle_threads_cv.wait( 
                    lock, [this] { return stop || !completed();});

                ++number_of_busy_threads;
                stats.busy_threads.update( number_of_busy_threads );

                --number_of_idle_threads;
                stats.idle_threads.update( number_of_idle_threads );
            }
        }
    }

    // require: scheduler_mutex is locked
    void activate_worker()
    {
        // do not activate a thread if there is nothing more to be done.
        if (completed())
            return;
        // otherwise activate a pending thread if there are any.
        else if (number_of_pending_threads)
            pending_threads_cv.notify_one();
        // otherwise activate an idle thread if there are any.
        else if (number_of_idle_threads)
            idle_threads_cv.notify_one();
        // otherwise start new thread if allowed.
        else if (thread_pool.size() < max_number_of_threads_totally)
        {
            ++number_of_busy_threads;
            stats.busy_threads.update( number_of_busy_threads ); 
                    
            thread_pool.emplace_back( std::jthread( &Scheduler::worker, this ));
        }
        // give up.
    }

    std::vector< std::jthread > thread_pool;

    const size_t max_number_of_busy_threads;
    const size_t max_number_of_threads_totally;
    std::mutex scheduler_mutex;
    std::condition_variable scheduler_cv;
    std::condition_variable pending_threads_cv;
    std::condition_variable idle_threads_cv;
    size_t number_of_busy_threads = 0;
    size_t number_of_blocked_threads = 0;
    size_t number_of_pending_threads = 0;
    size_t number_of_idle_threads = 0;
    SchedulerStats& stats;
    bool stop = false;
};