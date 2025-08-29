#pragma once

#include "player.h"
#include "game.h"
#include "node.h"

#include <random>
#include <thread>
#include <atomic>
#include <source_location>

namespace alphazero
{

namespace detail
{
    template< typename MoveT, typename StateT >
    struct Value
    {
        Value( Game< MoveT, StateT > game, MoveT const& move )
        : game( std::move(game) ), move( move ), game_result( this->game.result()) {}

        // Custom move constructor is needed because std::atomic is not movable
        // on all platforms. We "move" by loading the value from the source.
        Value(Value&& other) noexcept
            : game(std::move(other.game)),
              move(std::move(other.move)),
              game_result(other.game_result),
              visits(other.visits.load()),
              nn_policy(other.nn_policy),
              nn_value_sum(other.nn_value_sum.load())
        {}

        Value( Value const& ) = delete;
        Value& operator=( Value const& ) = delete;
        Value& operator=( Value&& ) = delete;
        ~Value() = default;

        Game< MoveT, StateT > game;
        MoveT move; // the previous move resulting in this game
        const GameResult game_result; // the cached game result

        std::atomic<size_t> visits;
        // policy probabilities (priors) of choosing this move
        float nn_policy = 0.0;
        std::atomic<float> nn_value_sum;
    };

    float game_result_2_score( GameResult, PlayerIndex );

} // namespace detail

template< typename MoveT, typename StateT >
using NodeAllocator = ::NodeAllocator< detail::Value< MoveT, StateT > >;

template< typename MoveT, typename StateT, size_t G, size_t P >
class Player : public ::Player< MoveT >
{
public:
    using game_type = Game< MoveT, StateT >;
    using value_type = detail::Value< MoveT, StateT >;
    using node_type = Node< value_type >;
    static constexpr std::size_t game_size = G;
    static constexpr std::size_t policy_size = P;

    Player(
        Game< MoveT, StateT > const& game,
        float c_base,
        float c_init,
        size_t simulations, // may be different from model training
        size_t opening_moves,
        unsigned seed,
        NodeAllocator< MoveT, StateT >& allocator,
        size_t threads )
    : root( new (allocator.allocate(1))
            node_type( value_type( game, MoveT()), allocator ),
            // The custom deleter now only needs to destruct and deallocate the single
            // node pointer it is given. The Node's destructor is responsible for
            // recursively cleaning up its children.
            [&allocator](node_type* ptr) {
                if (ptr) { ptr->~node_type(); allocator.deallocate(ptr, 1); } } ),
      c_base( c_base ), c_init( c_init ), simulations( simulations ),
      opening_moves( opening_moves ), g( seed ), allocator( allocator ),
      threads( threads )
    {
        if (!simulations)
            throw std::invalid_argument( "simulations must be > 0" );
        if (!threads)
            throw std::invalid_argument( "threads must be > 0" );
    }

    auto choose_move_iterator()
    {
        std::vector< std::jthread > thread_pool;
        thread_pool.reserve( threads );

        // The number of simulations to run in each thread.
        // Use ceiling division to ensure all simulations are run.
        const size_t sims_per_thread = (simulations + threads - 1) / threads;

        for (size_t i = 0; i < threads; ++i)
            thread_pool.emplace_back(
                [this, sims_per_thread]
                {
                    for (size_t j = 0; j < sims_per_thread; ++j)
                        simulation(*root);
                });

        for (auto& t : thread_pool)
            t.join();

        if (++move_count <= opening_moves)
            return choose_opening_move();
        else
            return choose_best_move();
    }

    MoveT choose_move() override
    {
        auto itr = choose_move_iterator();
        if (itr == root->get_children().end())
            throw std::source_location::current();

        auto new_root = &*itr;
        root->get_children().erase( itr );
        root.reset( new_root );

        return root->get_value().move;
    }

    NodePtr< value_type >& get_root() { return root; }
    size_t get_simulations() const { return simulations; }

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr = std::ranges::find_if(
            root->get_children(),
            [move](auto const& node)
                { return node.get_value().move == move; } );

        node_type* new_root = nullptr;
        if (itr == root->get_children().end())
            new_root = new
                (allocator.allocate(1))
                node_type( value_type(
                    root->get_value().game.apply( move ), move), allocator );
        else
        {
            new_root = &*itr;
            root->get_children().erase( itr );
        }

        root.reset( new_root );
    }

    // promise: return index of move in policy_vector
    virtual size_t move_to_policy_index( MoveT const& ) const = 0;

    float simulation( node_type& node )
    {
        auto lock = node.lock();
        auto& value = node.get_value();

        // Atomically update stats and apply virtual loss.
        // memory_order_relaxed is sufficient here because the mutexes already
        // provide the necessary memory ordering guarantees.
        const float virtual_loss = 1.0;
        value.visits.fetch_add(1, std::memory_order_relaxed);
        value.nn_value_sum.fetch_sub(virtual_loss, std::memory_order_relaxed);

        // set target value
        float nn_value;
        // from game result if game is decided
        if (value.game_result != GameResult::Undecided)
            nn_value = detail::game_result_2_score(
                value.game_result, value.game.current_player_index());
        else if (value.visits.load(std::memory_order_relaxed) == 1) // on first visit
        {
            expand( node );
            nn_value = nn_eval( node );
        }
        else
        // or simulate recursively otherwise
        // negate sign of return value because it's from the opponent's
        //  perspective
        {
            auto& child = select( node );
            lock.unlock();
            nn_value = -simulation( child );
        }

        lock.lock();
        // Atomically update value sum, removing virtual loss and adding the real result.
        value.nn_value_sum.fetch_add(virtual_loss + nn_value, std::memory_order_relaxed);

        return nn_value;
    }

    // require: node is not expanded
    void expand( node_type& node)
    {
        auto& value = node.get_value();
        for (MoveT const& move : value.game)
        {
            auto child = new
                (allocator.allocate(1))
                Node( value_type( value.game.apply( move ), move ), allocator );

            node.get_children().push_front( *child );
        }
    }

    // require: node is expanded
    float nn_eval( node_type& node )
    {
        auto& value = node.get_value();
        auto [nn_value, policies] = predict( serialize_state( value.game ));

        // convert logits of legal moves to probabilities with softmax
        // and save in children
        float policy_sum = 0.0f;
        for (auto& child : node.get_children())
        {
            const float p = std::expf( policies[
                move_to_policy_index( child.get_value().move )]);
            child.get_value().nn_policy = p;
            policy_sum += p;
        }
        // normalize
        for (auto& child : node.get_children())
            child.get_value().nn_policy /= policy_sum;

        return nn_value;
    }

    virtual std::array< float, G > serialize_state(
        Game< MoveT, StateT > const& ) const = 0;
protected:
    // upper confidence bound
    float ucb( value_type const& value, size_t parent_visits )
    {
        const float c =
            std::logf( (parent_visits + c_base + 1) / c_base) + c_init;

        // Atomically load visits and value_sum to get a consistent snapshot
        // for the UCB calculation. This prevents the data race where one thread
        // reads a new value_sum but an old visits count.
        size_t child_visits = value.visits.load(std::memory_order_relaxed);
        float child_value_sum = value.nn_value_sum.load(std::memory_order_relaxed);

        float q = 0.0;
        if (child_visits != 0)
            // switch sign because value is from the child's perspective
            q = -child_value_sum / static_cast< float >( child_visits );
        const float p = value.nn_policy;

        return q + c * p * std::sqrtf( static_cast< float >( parent_visits ))
            / static_cast< float >(child_visits + 1);
    }
    // Apple clang version 17.0.0 (clang-1700.0.13.3) produces strange errors
    // when i specify the return value iterator type explicitly:
    // Node< ValueT > member boost::intrusive::list< Node > children has
    // incomplete type
    // require: root node is expanded
    auto choose_best_move()
    {
        // choose child with most visits
        return std::ranges::max_element( root->get_children(),
            [](auto const& a, auto const& b)
            { return a.get_value().visits < b.get_value().visits; } );
    }
private:
    // require: root node is expanded
    auto choose_opening_move()
    {
        auto& children = root->get_children();

        // sample from children in opening phase by visit distribution
        // so we are more versatile in the opening
        size_t total_visits = 0;
        for (auto const& child : children)
            total_visits += child.get_value().visits.load(std::memory_order_relaxed);

        if (!total_visits)
            throw std::source_location::current();

        std::uniform_int_distribution< size_t > dist(0, total_visits - 1);
        size_t r = dist( g );
        size_t child_visits = 0;
        for (auto itr = children.begin(); itr != children.end(); ++itr)
        {
            child_visits = itr->get_value().visits.load(std::memory_order_relaxed);
            if (r < child_visits)
                return itr;
            r -= child_visits;
        }

        return children.begin(); // Fallback, should ideally not be reached
    }

    // require: node has children
    node_type& select( node_type& node )
    {
        return *std::ranges::max_element(
            node.get_children(),
            [this, parent_visits = node.get_value().visits.load()]
            (auto const& a, auto const& b)
            { return ucb( a.get_value(), parent_visits ) < ucb( b.get_value(), parent_visits ); });
    }

    // predict game state value and policy vector from nn
    // promise: returned policies contain probability distribution of moves
    virtual std::pair< float, std::array< float, P > > predict( std::array< float, G > const& ) = 0;

    NodePtr< value_type > root;
    float c_base;
    float c_init;
    const size_t simulations;
    const size_t opening_moves;
    size_t move_count = 0;
    std::mt19937 g;
    NodeAllocator< MoveT, StateT >& allocator;
    size_t threads;
};

namespace training {

template< size_t G, size_t P >
struct Position
{
    std::array< float, G > game_state_players;
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
        std::vector< Position< G, P >>& positions )
    : player( player ), dirichlet_epsilon( dirichlet_epsilon ),
      g( g ), gamma_dist( dirichlet_alpha, 1.0f ), positions( positions ) {}

    void run()
    {
        const size_t prev_size = positions.size();
        GameResult game_result;
        for (game_result = player.get_root()->get_value().game_result;
             game_result == GameResult::Undecided;
             game_result = player.get_root()->get_value().game_result)
        {
            auto& root = *player.get_root();

            // expand root node if first visited
            if (!root.get_value().visits)
            {
                player.expand( root );
                player.nn_eval( root );
            }

            add_dirichlet_noise();
            auto itr = player.choose_move_iterator();

            append_training_data();

            if (itr == root.get_children().end())
                throw std::source_location::current();
            auto new_root = &*itr;
            root.get_children().erase( itr );
            player.get_root().reset( new_root );
        }

        // set target value to game result for all new positions
        for (auto itr = positions.begin() + prev_size; itr != positions.end();
             ++itr)
            itr->target_value = detail::game_result_2_score(
                game_result, itr->current_player );
    }
protected:
    // require: root node is expanded
    void add_dirichlet_noise()
    {
        auto& root = *player.get_root();

        // add noise for root node
        for (auto& child : root.get_children())
        {
            float& policy = child.get_value().nn_policy;
            policy *= 1.0 - dirichlet_epsilon;
            policy += gamma_dist( g ) * dirichlet_epsilon;
        }
    }

    void append_training_data()
    {
        positions.emplace_back();
        auto& position = positions.back();
        auto& root = *player.get_root();
        auto& value = root.get_value();

        position.current_player = value.game.current_player_index();

        // append serialized game state
        position.game_state_players = player.serialize_state( value.game );

        size_t sum_visits = 0;
        for (auto const& child : root.get_children())
            sum_visits += child.get_value().visits;

        // append new target policies
        for (size_t policy_index = 0; policy_index < P; ++policy_index)
        {
            auto child_itr = std::ranges::find_if(
                root.get_children(),
                [this, policy_index](auto const& child)
                { return policy_index == player.move_to_policy_index( child.get_value().move ); } );
            if (child_itr == root.get_children().end())
                position.target_policy[policy_index] = 0.0f;
            else
                position.target_policy[policy_index] = static_cast< float >(
                    child_itr->get_value().visits ) / sum_visits;
        }
    }

    Player< MoveT, StateT, G, P >& player;
    float dirichlet_epsilon;
    std::mt19937& g;
    std::gamma_distribution< float > gamma_dist;
    std::vector< Position< G, P >>& positions;
};

} // namespace training
} // namespace alphazero