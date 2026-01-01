#pragma once

#include "exception.h"
#include "inference.h"
#include "node.h"
#include "player.h"
#include "statistics.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <mutex>
#include <random>

namespace alphazero
{

template < typename MoveT, typename StateT > struct Payload
{
    using node_type = Node< MoveT, StateT, Payload >;

    Payload() noexcept = default;
    Payload( Payload const &payload ) noexcept
        : policies_are_evaluated( payload.policies_are_evaluated.load() ),
          visits( payload.visits.load() ),
          nn_policy( payload.nn_policy.load() ),
          nn_value_sum( payload.nn_value_sum.load() )
    {
    }

    std::atomic< bool > policies_are_evaluated{ false };

    std::atomic< size_t > visits{ 0 };
    // policy probabilities of choosing this move from the parent's node
    // perspective.
    std::atomic< float > nn_policy{ 0.0f };
    // value from the current player's perspective.
    std::atomic< float > nn_value_sum{ 0.0f };
};

// the score's sign is depending on the player's perspective, thread-safe.
float game_result_2_score( GameResult, PlayerIndex );

namespace params
{

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
template < typename MoveT, typename StateT, typename PayloadT >
float shannon_entropy( Node< MoveT, StateT, PayloadT > const &node )
{
    size_t visit_sum = 0;
    for ( auto const &child : node.get_children() )
        visit_sum +=
            child.get_payload().visits.load( std::memory_order_relaxed );
    if ( !visit_sum )
        return 0.0f;

    float entropy = 0;
    for ( auto const &child : node.get_children() )
    {
        const size_t visits =
            child.get_payload().visits.load( std::memory_order_relaxed );
        if ( !visits )
            continue;
        const float p =
            static_cast< float >( visits ) / static_cast< float >( visit_sum );
        entropy -= p * std::logf( p );
    }
    return entropy;
}

namespace training
{
template < typename MoveT, typename StateT, size_t G, size_t P > class SelfPlay;
}

template < typename MoveT, typename StateT, size_t G, size_t P >
class Player : public ::Player< MoveT >
{
  public:
    using game_type = Game< MoveT, StateT >;
    using payload_type = Payload< MoveT, StateT >;
    using node_type = Node< MoveT, StateT, payload_type >;
    using pre_node_type = PreNode< MoveT, StateT, payload_type >;
    using fix_node_type = FixNode< MoveT, StateT, payload_type >;
    using inference_service_type = inference::Service< G, P >;
    using request_type = inference::Request< G, P >;
    using response_type = inference::Response< P >;
    using allocator_type = GenerationalArenaAllocator;
    using node_visitor_type = NodeVisitor< MoveT, StateT, payload_type >;

    static constexpr std::size_t game_size = G;
    static constexpr std::size_t policy_size = P;

    Player( game_type game, params::Ucb const &ucb,
            params::GamePlay const &game_play,
            unsigned seed, // make the play deterministic with seed
            allocator_type &allocator,
            inference_service_type &inference_service )
        : allocator( allocator ), ucb_params( ucb ), game_play( game_play ),
          g( seed ), thread_pool( game_play.parallel_simulations ),
          inference_service( inference_service ),
          response_queue( inference_service.get_max_batch_size() ),
          root( *( new ( allocator.allocate< pre_node_type >() )
                       pre_node_type( game ) ) )
    {
        if ( !game_play.simulations )
            throw beat_it::Exception( "Simulations cannot be zero." );
        if ( !game_play.parallel_simulations )
            throw beat_it::Exception( "Parallel simulations cannot be zero." );

        for ( auto &thread : thread_pool )
            thread =
                std::jthread( &Player::worker, this, stop_source.get_token() );
    }

    virtual ~Player() noexcept
    {
        stop_source.request_stop();
        // release all potentially waiting workers
        remaining_simulations.release( thread_pool.size() );
        for ( auto &thread : thread_pool )
            thread.join();
    }

    // not thread safe.
    MoveT choose_move() override
    {
        auto &node = run_simulations();
        allocator.reset();
        root = node.copy_tree( allocator );
        return root.get().get_move();
    }

    node_type &get_root() const noexcept { return root; }

    // require: move is valid, not thread safe.
    void apply_opponent_move( MoveT const &move ) override
    {
        auto &children = root.get().get_children();
        auto new_root_itr =
            std::ranges::find_if( children, [move]( auto const &node )
                                  { return node.get_move() == move; } );

        allocator.reset();
        if ( new_root_itr != children.end() )
        {
            root = new_root_itr->copy_tree( allocator );
            return;
        }

        auto *pre_node = dynamic_cast< pre_node_type * >( &root.get() );
        if ( pre_node )
        {
            root = *(
                new ( allocator.allocate< pre_node_type >() )
                    pre_node_type( pre_node->get_game().apply( move ), move ) );
            return;
        }

        throw beat_it::Exception( "Invalid move." );
    }

  protected:
    // promise: return index of move in policy_vector
    virtual size_t move_to_policy_index( MoveT const & ) const = 0;
    virtual std::array< float, G >
    serialize_state( game_type const & ) const = 0;

  private:
    friend class training::SelfPlay< MoveT, StateT, G, P >;

    struct ExpandVisitor : public node_visitor_type
    {
        explicit ExpandVisitor( Player &player ) noexcept : player( player ) {}

        // thread-safe.
        void visit( pre_node_type &node ) override
        {
            boost::intrusive::list< node_type > new_children;
            for ( MoveT const &move : node.get_game() )
            {
                auto &child = *(
                    new ( player.allocator.allocate< pre_node_type >() )
                        pre_node_type( node.get_game().apply( move ), move ) );

                // get write access lock.
                new_children.push_front( child );
            }

            std::scoped_lock _( node.get_mutex() );
            node.get_children().swap( new_children );
        }

        Player &player;
    };

    struct InferenceVisitor : public node_visitor_type
    {
        explicit InferenceVisitor( Player &player ) : player( player ) {}

        // thread-safe.
        void visit( pre_node_type &node ) noexcept override
        {
            // pushing a request to the inference service may block if the
            // request queue is full.
            player.inference_service.push(
                { .response_queue = &player.response_queue,
                  .cv = &player.simulations_cond,
                  .node = &node,
                  .state = player.serialize_state( node.get_game() ) } );
        }

        Player &player;
    };

    struct SelectVisitor : public node_visitor_type
    {
        explicit SelectVisitor( Player &player ) : player( player ) {}

        using base_node_type = Node< MoveT, StateT, payload_type >;

        // thread-safe.
        void update_selection_stats( base_node_type &node )
        {
            if ( node.get_payload().policies_are_evaluated.load(
                     std::memory_order_relaxed ) )
                player.informed_selections.fetch_add(
                    1, std::memory_order_relaxed );
            player.total_selections.fetch_add( 1, std::memory_order_relaxed );
        }

        // require: child has children.
        // not thread-safe.
        void select( base_node_type &node )
        {
            selected_node = &*std::ranges::max_element(
                node.get_children(),
                [this, parent_visits = node.get_payload().visits.load(
                           std::memory_order_relaxed )]( auto const &a,
                                                         auto const &b )
                {
                    return player.ucb( a.get_payload(), parent_visits ) <
                           player.ucb( b.get_payload(), parent_visits );
                } );
        }

        // thread-safe because fix nodes are expanded already.
        void visit( fix_node_type &node ) noexcept override
        {
            update_selection_stats( node );
            // fix nodes have children.
            select( node );
        }

        // thread-safe.
        void visit( pre_node_type &node ) noexcept override
        {
            update_selection_stats( node );

            while ( node.get_children().empty() )
                std::this_thread::yield();

            // multiple threads are allowed to read access, but it blocks if
            // the node is expanding right now.
            std::shared_lock _( node.get_mutex() );
            select( node );
        }

        Player &player;
        base_node_type *selected_node = nullptr;
    };

    // thread-safe.
    void simulation( node_type &node )
    {
        auto &payload = node.get_payload();

        // Atomically update visit count. on first visit expand leaf node.
        const bool is_leaf_node = payload.visits.fetch_add( 1 ) == 0;
        // apply virtual loss to avoid entering this path from the
        // parent's node perspective, so increase nn_value;
        payload.nn_value_sum.fetch_add( virtual_loss,
                                        std::memory_order_relaxed );

        if ( node.get_game_result() != GameResult::Undecided )
        {
            const float value = game_result_2_score(
                node.get_game_result(), node.get_current_player_index() );
            backpropagation( node, {}, value );
            // notify waiting thread because this update does not go through
            // nn evaluation.
            simulations_cond.notify_one();
        }
        else if ( is_leaf_node )
        {
            {
                ExpandVisitor visitor{ *this };
                node.accept( visitor );
            }

            InferenceVisitor visitor( *this );
            node.accept( visitor );
        }
        else
        {
            SelectVisitor visitor{ *this };
            node.accept( visitor );

            simulation( *visitor.selected_node );
        }
    }

    void process_response_queue()
    {
        response_type response;
        while ( response_queue.pop( response ) )
            backpropagation( *static_cast< node_type * >( response.node ),
                             response.policies, response.nn_value );
    }

    // require: single-threaded access.
    // return move to choose.
    node_type &run_simulations()
    {
        unfinished_simulations.store( game_play.simulations );
        // wake up waiting worker threads by initializing resource.
        remaining_simulations.release( game_play.simulations );

        // wait until all simulations finished.
        // note: wait with timeout because under rare conditions wait can
        // block forever. dead-lock scenario:
        // - no remaining simulations
        // - all simulation workers block in acquire
        // - inference service calculates some requests
        // - response queue is empty and no unfinished simulations
        // - cv wait condition is met but not entered blocking wait yet
        // - inference service pushes last responses and the cv notify is lost
        // - enter blocking cv wait
        // - cv will never be notified
        std::unique_lock lock( simulations_mutex );
        while ( unfinished_simulations.load() != 0 )
        {
            simulations_cond.wait_for(
                lock, std::chrono::milliseconds( 1000 ),
                [this]
                {
                    return !response_queue.empty() ||
                           unfinished_simulations.load() == 0;
                } );
            process_response_queue();
        }

        if ( ++move_count <= game_play.opening_moves )
            return choose_opening_move();
        else
            // choose child with most visits
            return *std::ranges::max_element(
                root.get().get_children(),
                []( auto const &a, auto const &b )
                {
                    return a.get_payload().visits.load(
                               std::memory_order_relaxed ) <
                           b.get_payload().visits.load(
                               std::memory_order_relaxed );
                } );
    }

    // upper confidence bound
    // thread-safe.
    float ucb( payload_type const &payload, size_t parent_visits )
    {
        const size_t child_visits =
            payload.visits.load( std::memory_order_relaxed );
        const float child_value_sum =
            payload.nn_value_sum.load( std::memory_order_relaxed );

        float c =
            static_cast< float >( parent_visits ) + ucb_params.c_base + 1.0f;
        c /= ucb_params.c_base;
        c = std::logf( c ) + ucb_params.c_init;
        c *= std::sqrtf( static_cast< float >( parent_visits ) );
        c /= static_cast< float >( child_visits + 1 );

        float q = 0.0;
        if ( child_visits != 0 )
            // switch sign because value is from the child's perspective
            q = -child_value_sum / static_cast< float >( child_visits );
        const float p = payload.nn_policy;

        return q + c * p;
    }

    // lockfree setting of policies and nn value and backpropagation.
    // require: mode is undecided.
    // thread-safe.
    void backpropagation( node_type &node,
                          std::array< float, P > const &policies,
                          float nn_value )
    {
        // convert logits of legal moves to probabilities with softmax
        // and save in children. the policies are guaranteed to be set only
        // once, so the lockfree code below is threadsafe.
        float policy_sum = 0.0f;
        std::array< float, P > policy_buffer;
        auto p_itr = policy_buffer.begin();
        for ( node_type &child : node.get_children() )
        {
            const float p =
                std::expf( policies[move_to_policy_index( child.get_move() )] );
            *p_itr++ = p;
            policy_sum += p;
        }

        // normalize priors to probabilities.
        p_itr = policy_buffer.begin();
        for ( auto &child : node.get_children() )
            child.get_payload().nn_policy.store( *p_itr++ / policy_sum,
                                                 std::memory_order_relaxed );

        node.get_payload().policies_are_evaluated.store(
            true, std::memory_order_relaxed );

        // remove virtual loss.
        node.get_payload().nn_value_sum.fetch_sub( virtual_loss,
                                                   std::memory_order_relaxed );

        // backpropagate nn value. note the alternating sign of nn_value
        // the player's perspective is changing on the way up to the root node.
        for ( node_type *next = &node; next; next = next->get_parent() )
        {
            next->get_payload().nn_value_sum.fetch_add(
                nn_value, std::memory_order_relaxed );
            // toggle sign
            nn_value = -nn_value;
        }

        unfinished_simulations.fetch_sub( 1 );
    }

    // thread-safe.
    void worker( std::stop_token token )
    {
        while ( true )
        {
            remaining_simulations.acquire();
            if ( token.stop_requested() )
                break;
            simulation( root );
        }
    }

    // require: root node is expanded
    node_type &choose_opening_move()
    {
        auto &children = root.get().get_children();

        // sample from children in opening phase by visit distribution
        // so we are more versatile in the opening
        size_t total_visits = 0;
        for ( auto const &child : children )
            total_visits +=
                child.get_payload().visits.load( std::memory_order_relaxed );

        if ( !total_visits )
        {
            std::uniform_int_distribution< size_t > dist(
                0, std::size( children ) - 1 );
            return *std::next( children.begin(), dist( g ) );
        }

        std::uniform_int_distribution< size_t > dist( 0, total_visits - 1 );
        size_t r = dist( g );
        size_t child_visits = 0;
        for ( auto itr = children.begin(); itr != children.end(); ++itr )
        {
            child_visits =
                itr->get_payload().visits.load( std::memory_order_relaxed );
            if ( r < child_visits )
                return *itr;
            r -= child_visits;
        }

        throw beat_it::Exception( "cannot choose move" );
    }

    allocator_type &allocator;
    const params::Ucb ucb_params;
    const params::GamePlay game_play;
    size_t move_count = 0;
    std::mt19937 g;
    std::vector< std::jthread > thread_pool;
    std::counting_semaphore<> remaining_simulations{ 0 };
    std::atomic< size_t > unfinished_simulations{ 0 };
    std::mutex simulations_mutex;
    std::condition_variable simulations_cond;
    std::stop_source stop_source;
    inference_service_type &inference_service;
    const float virtual_loss = 1.0; // must be positive
    boost::lockfree::queue< response_type > response_queue;
    std::reference_wrapper< node_type > root;
    std::atomic< size_t > informed_selections{ 0 };
    std::atomic< size_t > total_selections{ 0 };
};

namespace training
{

template < size_t G, size_t P > struct Position
{
    std::array< float, G > game_state;
    std::array< float, P > target_policy;
    float target_value = 0.0f;
    PlayerIndex current_player;
};

template < typename MoveT, typename StateT, size_t G, size_t P > class SelfPlay
{
  public:
    using player_type = Player< MoveT, StateT, G, P >;
    using expand_visitor_type = player_type::ExpandVisitor;
    using pre_node_type = player_type::pre_node_type;

    SelfPlay( player_type &player, float dirichlet_alpha,
              float dirichlet_epsilon, std::mt19937 &g,
              std::vector< Position< G, P > > &positions,
              Statistics &root_node_entropy_stat,
              Statistics &informed_selection_stats ) noexcept
        : player( player ), dirichlet_epsilon( dirichlet_epsilon ), g( g ),
          gamma_dist( dirichlet_alpha, 1.0f ), positions( positions ),
          root_node_entropy_stat( root_node_entropy_stat ),
          informed_selection_stats( informed_selection_stats )
    {
    }

    Statistics const &get_root_node_entropy() const noexcept
    {
        return root_node_entropy_stat;
    }

    void run()
    {
        const size_t prev_size = positions.size();
        GameResult game_result;
        // we need to keep track of the actual game because it will not be
        // copied in the next generation root node due to optimization.
        Game< MoveT, StateT > game = [&root = player.get_root()]
        {
            auto *pre_node = dynamic_cast< pre_node_type * >( &root );
            if ( !pre_node )
                throw beat_it::Exception( "invalid root node type" );
            return pre_node->get_game();
        }();

        for ( game_result = player.get_root().get_game_result();
              game_result == GameResult::Undecided;
              game_result = player.get_root().get_game_result() )
        {
            auto &root = player.get_root();
            // expand root node on first visit, for dirichlet noise addition
            // the root node needs to be expanded.
            if ( root.get_children().empty() )
            {
                expand_visitor_type visitor{ player };
                player.get_root().accept( visitor );
            }
            add_dirichlet_noise();

            auto &node = player.run_simulations();
            append_training_data( game );
            // update game.
            game = game.apply( node.get_move() );

            root_node_entropy_stat.update( shannon_entropy( root ) );

            informed_selection_stats.update(
                static_cast< float >( player.informed_selections.load(
                    std::memory_order_relaxed ) ) /
                static_cast< float >( player.total_selections.load(
                    std::memory_order_relaxed ) ) );
            player.informed_selections.store( 0, std::memory_order_relaxed );
            player.total_selections.store( 0, std::memory_order_relaxed );

            player.allocator.reset();
            player.root = node.copy_tree( player.allocator );
        }

        // set target value to game result for all new positions
        for ( auto itr = positions.begin() + prev_size; itr != positions.end();
              ++itr )
            itr->target_value =
                game_result_2_score( game_result, itr->current_player );
    }

  private:
    // require: root node is expanded
    void add_dirichlet_noise()
    {
        auto &root = player.get_root();

        // add noise for root node
        for ( auto &child : root.get_children() )
        {
            float policy =
                child.get_payload().nn_policy.load( std::memory_order_relaxed );
            policy *= 1.0f - dirichlet_epsilon;
            policy += gamma_dist( g ) * dirichlet_epsilon;
            child.get_payload().nn_policy.store( policy,
                                                 std::memory_order_relaxed );
        }
    }

    void append_training_data( Game< MoveT, StateT > const &game )
    {
        positions.emplace_back();
        auto &position = positions.back();
        auto &root = player.get_root();
        auto &payload = root.get_payload();

        position.current_player = root.get_current_player_index();

        // append serialized game state
        position.game_state = player.serialize_state( game );

        // root visits should be the children's visits sum.
        const size_t sum_visits =
            payload.visits.load( std::memory_order_relaxed );

        // Initialize all policies to zero.
        position.target_policy.fill( 0.0f );

        // Then, iterate through the children once to populate the policies.
        // This is more efficient than iterating through all possible moves.
        if ( sum_visits > 0 )
            for ( auto const &child : root.get_children() )
            {
                const size_t policy_index =
                    player.move_to_policy_index( child.get_move() );
                position.target_policy[policy_index] =
                    static_cast< float >( child.get_payload().visits.load(
                        std::memory_order_relaxed ) ) /
                    sum_visits;
            }
    }

  private:
    Player< MoveT, StateT, G, P > &player;
    float dirichlet_epsilon;
    std::mt19937 &g;
    std::gamma_distribution< float > gamma_dist;
    std::vector< Position< G, P > > &positions;
    Statistics &root_node_entropy_stat;
    Statistics &informed_selection_stats;
};

} // namespace training
} // namespace alphazero
