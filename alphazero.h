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

        const Game< MoveT, StateT > game;
        const MoveT move; // the previous move resulting in this game
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

namespace params {

struct Ucb
{
    float c_base = 0.0f;
    float c_init = 0.0f;
};

struct GamePlay
{
    // simulations and opening moves may be different from model training
    size_t simulations = 0;
    size_t opening_moves = 0;
    size_t threads = 0;
};

} // namespace params

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
        Game< MoveT, StateT > game,
        params::Ucb const& ucb,
        params::GamePlay const& game_play,
        unsigned seed, // make the play deterministic with seed
        NodeAllocator< MoveT, StateT >& allocator)
    : root( new (allocator.allocate(1)) node_type(
                value_type( std::move(game), MoveT()), allocator ),
            NodeDeleter<value_type>{allocator} ),
      ucb_params( ucb ), game_play( game_play ), g( seed ),
      allocator( allocator )
    {
        if (!game_play.simulations)
            throw std::source_location::current();
        if (!game_play.threads)
            throw std::source_location::current();
    }

    auto choose_move_iterator()
    {
        std::vector< std::jthread > thread_pool;
        thread_pool.reserve( game_play.threads );

        // The number of simulations to run in each thread.
        // Use ceiling division to ensure all simulations are run.
        const size_t sims_per_thread =
            (game_play.simulations + game_play.threads - 1) / game_play.threads;

        for (size_t i = 0; i < game_play.threads; ++i)
            thread_pool.emplace_back(
                [this, sims_per_thread]
                {
                    for (size_t j = 0; j < sims_per_thread; ++j)
                        simulation(*root);
                });

        // destroy and join all jthreads
        thread_pool.clear();

        if (++move_count <= game_play.opening_moves)
            return choose_opening_move();
        else
            return choose_best_move();
    }

    MoveT choose_move() override
    {
        auto itr = choose_move_iterator();
        advance_root(itr);
        return root->get_value().move;
    }

    // Advances the root of the MCTS tree to the specified child node.
    void advance_root(auto itr)
    {
        if (itr == root->get_children().end())
            throw std::source_location::current();
        // Get the pointer to the new root node BEFORE erasing it from the list.
        node_type* new_root = &*itr;
        // Erasing the node from the list invalidates the iterator `itr`.
        root->get_children().erase(itr);
        // Reset the unique_ptr to manage the new root. The old root and its
        // other children are destroyed automatically.
        root.reset(new_root);
    }

    NodePtr< value_type >& get_root() { return root; }
    size_t get_simulations() const { return game_play.simulations; }

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
        auto& value = node.get_value();

        // Atomically update visit count and apply virtual loss.
        // 1.0 means assuming having lost this position.
        const float virtual_loss = 1.0;
        value.visits.fetch_add( 1, std::memory_order_relaxed);
        value.nn_value_sum.fetch_sub( virtual_loss, std::memory_order_relaxed);

        // set target value
        float nn_value;
        // from game result if game is decided
        if (value.game_result != GameResult::Undecided)
            nn_value = detail::game_result_2_score(
                value.game_result, value.game.current_player_index());
        else
        {
            // The lock is crucial to ensure that only one thread can perform
            // the check-and-expand operation on a leaf node.
            std::unique_lock lock(node.lock());

            // A node is a leaf if it has no children. The first thread to
            // arrive will expand it. This check is now race-free.
            if (node.get_children().empty())
            {
                expand( node );
                // Pass the lock to nn_eval, which will temporarily unlock it
                // during the slow network prediction.
                nn_value = nn_eval( node, &lock );
            }
            else // otherwise simulate recursively otherwise
            {
                auto& child = select( node );
                // unlock the node before simulate recursively
                lock.unlock();
                // negate sign of return value because it's from the opponent's
                //  perspective
                nn_value = -simulation( child );
            }
        }

        // Atomically remove virtual loss and add the real result.
        value.nn_value_sum.fetch_add(
            virtual_loss + nn_value, std::memory_order_relaxed);

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
    float nn_eval(
        node_type& node, std::unique_lock< std::mutex >* lock = nullptr )
    {
        auto& value = node.get_value();

        // Temporarily unlock the node during the potentially slow neural
        // network evaluation to allow other threads to continue their search.
        if (lock)
            lock->unlock();
        auto [nn_value, policies] = predict( serialize_state( value.game ));
        if (lock) // Re-acquire the lock to safely update children.
            lock->lock();

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
            std::logf((parent_visits + ucb_params.c_base + 1)
                / ucb_params.c_base) + ucb_params.c_init;

        // Atomically load visits and value_sum to get a consistent snapshot
        // for the UCB calculation. This prevents the data race where one thread
        // reads a new value_sum but an old visits count.
        size_t child_visits = value.visits.load(std::memory_order_relaxed);
        float child_value_sum = value.nn_value_sum.load(
            std::memory_order_relaxed);

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
            total_visits += child.get_value().visits.load(
                std::memory_order_relaxed);

        if (!total_visits)
            throw std::source_location::current();

        std::uniform_int_distribution< size_t > dist(0, total_visits - 1);
        size_t r = dist( g );
        size_t child_visits = 0;
        for (auto itr = children.begin(); itr != children.end(); ++itr)
        {
            child_visits = itr->get_value().visits.load(
                std::memory_order_relaxed);
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
            {
                return   ucb( a.get_value(), parent_visits )
                       < ucb( b.get_value(), parent_visits );
            });
    }

    // predict game state value and policy vector from nn
    // promise: returned policies contain probability distribution of moves
    virtual std::pair< float, std::array< float, P > >
        predict( std::array< float, G > const& ) = 0;

    NodePtr< value_type > root;
    const params::Ucb ucb_params;
    const params::GamePlay game_play;
    size_t move_count = 0;
    std::mt19937 g;
    NodeAllocator< MoveT, StateT >& allocator;
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
            // expand root node on first visit, for dirichlet noise addition
            //  the root node needs to be expanded and evaluated
            if (auto& root = *player.get_root(); !root.get_value().visits)
            {
                player.expand( root );
                player.nn_eval( root );
            }
            add_dirichlet_noise();

            auto itr = player.choose_move_iterator();

            append_training_data();

            player.advance_root(itr);
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
            policy *= 1.0f - dirichlet_epsilon;
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
        for (auto const& child : root.get_children()) {
            sum_visits += child.get_value().visits.load(std::memory_order_relaxed);
        }

        // Initialize all policies to zero.
        position.target_policy.fill(0.0f);

        // Then, iterate through the children once to populate the policies.
        // This is more efficient than iterating through all possible moves.
        if (sum_visits > 0) {
            for (auto const& child : root.get_children()) {
                const size_t policy_index = player.move_to_policy_index(child.get_value().move);
                position.target_policy[policy_index] =
                    static_cast<float>(child.get_value().visits.load(std::memory_order_relaxed)) / sum_visits;
            }
        }
    }
private:
    Player< MoveT, StateT, G, P >& player;
    float dirichlet_epsilon;
    std::mt19937& g;
    std::gamma_distribution< float > gamma_dist;
    std::vector< Position< G, P >>& positions;
};

} // namespace training
} // namespace alphazero