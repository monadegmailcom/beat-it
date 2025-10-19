#pragma once

#include "player.h"
#include "game.h"
#include "node.h"
#include "inference.h"
#include "exception.h"
#include <cassert>
#include <random>
#include <thread>
#include <semaphore>
#include <condition_variable>
#include <atomic>
#include <source_location>
#include <ranges>
#include <algorithm>
#include <iostream>
#include <system_error>

namespace alphazero
{

// additional payload for undecided leaf nodes. for inner 
// game nodes or decided games we do not need the additional payload. for 
// such nodes the additional playload will not be copied on node allocator 
// swapping.
template< typename MoveT, typename StateT >
struct AdditionalPayload
{
    const Game< MoveT, StateT > game;
    std::shared_mutex node_mutex;
};

template< typename MoveT, typename StateT >
struct Value
{
    using add_payload_type = AdditionalPayload< MoveT, StateT >;

    Value( 
        add_payload_type& additional_payload, MoveT const& move,
        Node< Value >* parent ) noexcept : 
        move( move ), game_result( additional_payload.game.result()), 
        current_player_index( additional_payload.game.current_player_index()),
        parent( parent ), additional_payload( &additional_payload ) {}

    // note: additional payload is not copied!
    Value( Value const& value ) noexcept : 
        move( value.move ), game_result( value.game_result ), 
        current_player_index( value.current_player_index ),
        parent( value.parent ), visits( value.visits.load()), 
        nn_policy( value.nn_policy.load()),
        nn_value_sum( value.nn_value_sum.load()) {}

    const MoveT move; // the previous move resulting in this game
    // game_result and current_player_index has be saved as attributes because
    // additional_payload may not be present.
    const GameResult game_result;
    const PlayerIndex current_player_index;
    // for backpropagation we need the parent node, may be null for root.
    Node< Value >* const parent;
    // additional payload for undecided leaf nodes.
    AdditionalPayload< MoveT, StateT >* additional_payload = nullptr;

    // a node is expanded iff its visits count is > 0.
    std::atomic< size_t > visits { 0 };
    // policy probabilities of choosing this move from the parent's node
    // perspective.
    std::atomic< float > nn_policy { 0.0f };
    // value from the current player's perspective.
    std::atomic< float > nn_value_sum { 0.0f };
};

// not thread-safe.
template< typename MoveT, typename StateT >
bool is_fixed( Node< Value< MoveT, StateT >> const& node )
{
    auto const& value = node.get_value();
    return value.game_result == GameResult::Undecided
        && value.visits.load( std::memory_order_relaxed);
}

// the score's sign is depending on the player's perspective, thread-safe.
float game_result_2_score( GameResult, PlayerIndex );

namespace params {

struct Ucb
{
    float c_base = 0.0f;
    float c_init = 0.0f;
};

struct GamePlay
{
    // simulations and opening moves may be different from model training.
    size_t simulations = 0;
    size_t opening_moves = 0;
    size_t parallel_simulations = 0;
};

} // namespace params

// not thread-safe. 
template< typename ValueT >
float shannon_entropy( Node< ValueT > const& node )
{
    size_t visit_sum = 0;
    for (auto const& child : node.get_children())
        visit_sum += child.get_value().visits.load( std::memory_order_relaxed );
    if (!visit_sum)
        return 0.0f;

    float entropy = 0;
    for (auto const& child : node.get_children())
    {
        const size_t visits = child.get_value().visits.load(
            std::memory_order_relaxed );
        if (!visits)
            continue;
        const float p = static_cast< float >( visits ) / 
                        static_cast< float >( visit_sum );
        entropy -= p * std::logf( p );
    }
    return entropy;
}

namespace training {
template< typename MoveT, typename StateT, size_t G, size_t P >
class SelfPlay;
}

template< typename MoveT, typename StateT, size_t G, size_t P >
class Player : public ::Player< MoveT >
{
public:
    using game_type = Game< MoveT, StateT >;
    using value_type = Value< MoveT, StateT >;
    using node_type = Node< value_type >;
    using inference_service_type = inference::Service< G, P >;
    using request_type = inference::Request< G, P >;
    using response_type = inference::Response< P >;
    using allocator_type = GenerationalArenaAllocator;
    using additional_payload_type = AdditionalPayload< MoveT, StateT >;

    static constexpr std::size_t game_size = G;
    static constexpr std::size_t policy_size = P;

    Player(
        Game< MoveT, StateT > game, params::Ucb const& ucb,
        params::GamePlay const& game_play,
        unsigned seed, // make the play deterministic with seed
        allocator_type& allocator, 
        inference_service_type& inference_service ) : 
        allocator( allocator ), ucb_params( ucb ), game_play( game_play ), 
        g( seed ), thread_pool( game_play.parallel_simulations ),
        inference_service( inference_service ),
        response_queue( inference_service.get_max_batch_size()),
        root( build_node( game, MoveT(), nullptr ))
    {
        if (!game_play.simulations)
            throw beat_it::Exception("Simulations cannot be zero.");
        if (!game_play.parallel_simulations)
            throw beat_it::Exception("Parallel simulations cannot be zero.");

        for (auto& thread : thread_pool)
            thread = std::jthread( 
                &Player::worker, this, stop_source.get_token());
    }

    virtual ~Player() noexcept
    {
        stop_source.request_stop();
    }

    // not thread safe.
    MoveT choose_move() override
    {
        auto& node = run_simulations();
        allocator.reset(); 
        root = copy_tree( node );
        return root.get().get_value().move;
    }

    node_type& get_root() const { return root; }

    // require: move is valid.
    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr = std::ranges::find_if(
            root.get().get_children(),
            [move](auto const& node)
                { return node.get_value().move == move; } );
        if (itr == root.get().get_children().end())
            throw std::source_location::current();
        node_type& new_root = *itr; 
        allocator.reset();
        root = copy_tree( new_root );
    }
protected:
    // promise: return index of move in policy_vector
    virtual size_t move_to_policy_index( MoveT const& ) const = 0;
    virtual std::array< float, G > serialize_state(
        game_type const& ) const = 0;
private:
    friend class training::SelfPlay< MoveT, StateT, G, P >;

    node_type& build_node( 
        game_type const& game, MoveT const& move, node_type* parent)
    {
        auto* additional_payload = 
            new (allocator.allocate< additional_payload_type >()) 
                additional_payload_type { .game = game };
        auto* node = new (allocator.allocate< node_type >()) 
            node_type( value_type( *additional_payload, move, parent ));
        return *node;
    }

    node_type& copy_tree( node_type const& node )
    {
        // copy node, note: only copy additional payload if necessary.
        auto* new_node = new (allocator.allocate< node_type >()) 
            node_type( node.get_value());
        if (is_fixed( node ))
            new_node->get_value().additional_payload = 
                new (allocator.allocate< additional_payload_type >())
                    additional_payload_type 
                    { .game = node.get_value().additional_payload->game };

        for (node_type const& child : node.get_children())
            new_node->get_children().push_back( copy_tree( child ));
        return *new_node;
    }

    // require: node is not expanded and game is Undecided.
    void expand( node_type& node)
    {
        auto& value = node.get_value();
        if (!value.additional_payload)
            throw std::source_location::current();
        auto& game = value.additional_payload->game;

        for (MoveT const& move : game)
        {
            auto& child = build_node( game.apply( move ), move, &node );

            std::unique_lock lock(value.additional_payload->node_mutex);
            node.get_children().push_front( child );
        }
    }

    void simulation( node_type& node )
    {
        auto& value = node.get_value();

        // Atomically update visit count. on first visit expand leaf node.
        const bool is_leaf_node = 
            value.visits.fetch_add( 1, std::memory_order_relaxed) == 0;
        // and apply virtual loss to avoid entering this path from the 
        // parent's node perspective, so increase nn_value;
        value.nn_value_sum.fetch_add( virtual_loss, std::memory_order_relaxed);

        if (value.game_result != GameResult::Undecided)
            backpropagation( 
                node, {}, game_result_2_score(
                    value.game_result, value.current_player_index)); 
        else if (is_leaf_node)
        {
            expand( node );
            if (!value.additional_payload)
                throw std::source_location::current();

            // pushing a request to the inference service may block if the
            // request queue is full.
            inference_service.push( {
                .response_queue = &response_queue,
                .node = &node,
                .state = serialize_state( value.additional_payload->game )
            });
        }
        else
            simulation( select( node ));
    }

    // require: single-threaded access.
    // return move to choose.
    node_type& run_simulations()
    {        
        // wake up waiting worker threads.
        remaining_simulations.fetch_add( 
            game_play.simulations, std::memory_order_relaxed);
        simulations_cv.notify_all();

        // goto sleep until all simulations finished.
        std::unique_lock lock( simulations_mutex );
        simulations_cv.wait(
            lock, stop_source.get_token(), 
            [this]
            {
                return remaining_simulations.load( 
                    std::memory_order_relaxed) == 0; 
            });

        if (++move_count <= game_play.opening_moves)
            return choose_opening_move();
        else
            return choose_best_move();
    }

    // upper confidence bound
    float ucb( value_type const& value, size_t parent_visits )
    {
        const size_t child_visits = value.visits.load(
            std::memory_order_relaxed);
        const float child_value_sum = value.nn_value_sum.load(
            std::memory_order_relaxed);

        float c = 
            static_cast< float >( parent_visits ) + ucb_params.c_base + 1.0f;
        c /= ucb_params.c_base;
        c = std::logf( c ) + ucb_params.c_init;
        c *= std::sqrtf( static_cast< float >( parent_visits ));
        c /= static_cast< float >(child_visits + 1);

        float q = 0.0;
        if (child_visits != 0)
            // switch sign because value is from the child's perspective
            q = -child_value_sum / static_cast< float >( child_visits );
        const float p = value.nn_policy;

        return q + c * p;
    }

    // require: root node is expanded
    node_type& choose_best_move()
    {
        // choose child with most visits
        return *std::ranges::max_element( 
            root.get().get_children(),
            [](auto const& a, auto const& b)
            { 
                return a.get_value().visits.load( std::memory_order_relaxed ) 
                    < b.get_value().visits.load( std::memory_order_relaxed );    
            });
    }

    // lockfree setting of policies and nn value and backpropagation.
    // require: mode is undecided.
    void backpropagation( 
        node_type& node, std::array< float, P > const& policies, 
        float nn_value )
    {
        value_type& value = node.get_value();
       
        // convert logits of legal moves to probabilities with softmax
        // and save in children. the policies are guaranteed to be set only
        // once, so the lockfree code below is threadsafe.

        float policy_sum = 0.0f;
        policy_buffer.clear();
        for (node_type& child : node.get_children())
        {
            const float p = std::expf( policies[
                move_to_policy_index( child.get_value().move )]);
            policy_buffer.push_back( p );
            policy_sum += p;
        }

        // normalize priors to probabilities.
        auto p_itr = policy_buffer.begin();
        for (auto& child : node.get_children())
        { 
            const float p = *p_itr++ / policy_sum;
            child.get_value().nn_policy.store( 
                p, std::memory_order_relaxed ); 
        }

        // remove virtual loss.
        value.nn_value_sum.fetch_sub( virtual_loss, std::memory_order_relaxed);

        // backpropagate nn value. note the alternating sign of nn_value
        // the player's perspective is changing on the way up to the root node.
        for (node_type* next = &node; next; next = next->get_value().parent)
        { 
            next->get_value().nn_value_sum.fetch_add(
                nn_value, std::memory_order_relaxed);
            // toggle sign
            nn_value = -nn_value;
        }

        // notify all potential waiter if simulation is completed.
        if (remaining_simulations.fetch_sub(1, std::memory_order_relaxed) == 1)
            simulations_cv.notify_all();                   
    }

    void worker( std::stop_token token )
    {
        while (!token.stop_requested())
        {
            // first backpropagate nn responses.
            response_type response;
            while (response_queue.pop( response ))
                backpropagation( 
                    *static_cast< node_type* >( response.node ), 
                    response.policies, response.nn_value );

            if (remaining_simulations.load( std::memory_order_relaxed) != 0)
                // then simulate. 
                simulation( root );
            else
            {
                // blocking wait if all simulations are finished.
                // wake up if simulations are pending again.
                std::unique_lock lock( simulations_mutex );
                simulations_cv.wait( 
                    lock, token,
                    [this]
                    { 
                        return remaining_simulations.load( 
                            std::memory_order_relaxed) != 0; 
                    });
            }
        }
    }

    // require: root node is expanded
    node_type& choose_opening_move()
    {
        auto& children = root.get().get_children();

        // sample from children in opening phase by visit distribution
        // so we are more versatile in the opening
        size_t total_visits = 0;
        for (auto const& child : children)
            total_visits += child.get_value().visits.load(
                std::memory_order_relaxed);

        if (!total_visits)
        {
            std::uniform_int_distribution< size_t > dist(
                0, std::size( children ) - 1);
            return *std::next( children.begin(), dist( g ));
        }

        std::uniform_int_distribution< size_t > dist(0, total_visits - 1);
        size_t r = dist( g );
        size_t child_visits = 0;
        for (auto itr = children.begin(); itr != children.end(); ++itr)
        {
            child_visits = itr->get_value().visits.load(
                std::memory_order_relaxed);
            if (r < child_visits)
                return *itr;
            r -= child_visits;
        }

        throw std::source_location::current(); 
    }

    // require: node has children
    node_type& select( node_type& node )
    {
        // multiple threads are allowed to read access, but it blocks if
        // node is in expanding right now.
        auto& value = node.get_value();
        if (!value.additional_payload)
            throw std::source_location::current();
        std::shared_lock lock( value.additional_payload->node_mutex );
        return *std::ranges::max_element(
            node.get_children(),
            [this, parent_visits = value.visits.load( 
                                    std::memory_order_relaxed)]
            (auto const& a, auto const& b)
            {
                return   ucb( a.get_value(), parent_visits )
                       < ucb( b.get_value(), parent_visits );
            });
    }

    allocator_type& allocator;
    const params::Ucb ucb_params;
    const params::GamePlay game_play;
    size_t move_count = 0;
    std::mt19937 g;
    std::vector< std::jthread > thread_pool;
    std::atomic< size_t > remaining_simulations {0};
    std::condition_variable_any simulations_cv;
    std::mutex simulations_mutex;
    std::stop_source stop_source; 
    inference_service_type& inference_service;
    const float virtual_loss = 1.0; // must be positive
    std::vector< float > policy_buffer; 
    boost::lockfree::queue< response_type > response_queue;
    std::reference_wrapper< node_type > root;
};

namespace training {

template< size_t G, size_t P >
struct Position
{
    std::array< float, G > game_state;
    std::array< float, P > target_policy;
    float target_value = 0.0f;
    PlayerIndex current_player;
};

template< typename MoveT, typename StateT, size_t G, size_t P >
class SelfPlay
{
public:
    SelfPlay(
        Player< MoveT, StateT, G, P >& player,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        std::mt19937& g,
        std::vector< Position< G, P >>& positions,
        Statistics& root_node_entropy_stat )
    : player( player ), dirichlet_epsilon( dirichlet_epsilon ),
      g( g ), gamma_dist( dirichlet_alpha, 1.0f ), positions( positions ),
      root_node_entropy_stat( root_node_entropy_stat ) {}

    Statistics const& get_root_node_entropy() const noexcept 
    { return root_node_entropy_stat; }

    void run()
    {
        const size_t prev_size = positions.size();
        GameResult game_result;
        // we need to keep track of the actual game because it will not be
        // copied in the next generation root node due to optimization.
        Game< MoveT, StateT > game = 
            player.get_root().get_value().additional_payload->game;
        for (game_result = player.get_root().get_value().game_result;
             game_result == GameResult::Undecided;
             game_result = player.get_root().get_value().game_result)
        {
            // expand root node on first visit, for dirichlet noise addition
            //  the root node needs to be expanded
            if (auto& root = player.get_root(); !root.get_value().visits)
                player.expand( root );

            add_dirichlet_noise();

            auto& node = player.run_simulations();
            append_training_data( game );
            // update game.
            game = game.apply( node.get_value().move );

            root_node_entropy_stat.update(
                shannon_entropy( player.get_root() ));

            player.allocator.reset();
            player.copy_tree( node );
        }

        // set target value to game result for all new positions
        for (auto itr = positions.begin() + prev_size; itr != positions.end();
             ++itr)
            itr->target_value = game_result_2_score(
                game_result, itr->current_player );
    }
protected:
    // require: root node is expanded
    void add_dirichlet_noise()
    {
        auto& root = player.get_root();

        // add noise for root node
        for (auto& child : root.get_children())
        {
            float policy = child.get_value().nn_policy.load(
                std::memory_order_relaxed);
            policy *= 1.0f - dirichlet_epsilon;
            policy += gamma_dist( g ) * dirichlet_epsilon;
            child.get_value().nn_policy.store( 
                policy, std::memory_order_relaxed );
        }
    }

    void append_training_data( Game< MoveT, StateT > const& game )
    {
        positions.emplace_back();
        auto& position = positions.back();
        auto& root = player.get_root();
        auto& value = root.get_value();

        position.current_player = value.current_player_index;

        // append serialized game state
        position.game_state = player.serialize_state( game );

        // root visits should be the children's visits sum.
        const size_t sum_visits = root.get_value().visits.load(
            std::memory_order_relaxed);

        // Initialize all policies to zero.
        position.target_policy.fill(0.0f);

        // Then, iterate through the children once to populate the policies.
        // This is more efficient than iterating through all possible moves.
        if (sum_visits > 0) 
            for (auto const& child : root.get_children()) 
            {
                const size_t policy_index = player.move_to_policy_index(
                    child.get_value().move);
                position.target_policy[policy_index] =
                    static_cast<float>(child.get_value().visits.load(
                        std::memory_order_relaxed)) / sum_visits;
            }
    }
private:
    Player< MoveT, StateT, G, P >& player;
    float dirichlet_epsilon;
    std::mt19937& g;
    std::gamma_distribution< float > gamma_dist;
    std::vector< Position< G, P >>& positions;
    Statistics& root_node_entropy_stat;
};

} // namespace training
} // namespace alphazero