#pragma once

namespace multiplayer {

namespace mcts {

namespace detail {

template< typename MoveT, typename PlayerIndexT, typename StateT >
struct Value
{
    using game_type = Game< MoveT, PlayerIndexT, StateT >;

    Value( game_type const& game, MoveT const& move )
    : game( game ), move( move ), game_result( game.result()),
      next_move_itr(this->game.begin()) {}

    Value( Value&& other ) noexcept
        : game(std::move(other.game)), // Game is truly moved
          move(std::move(other.move)),
          game_result(other.game_result),
          next_move_itr(this->game.begin()), // Iterator bound to the newly moved-to this->game
          points(other.points),
          visits(other.visits) {}

    game_type game;
    MoveT move; // the previous move resulting in this game
    const GameResult game_result; // the cached game result
    // iterator to next valid move not already added as a child node
    typename game_type::MoveItr next_move_itr;
    double points = 0.0; // 1 for win, 0.5 for draw, 0 for loss
    size_t visits = 0;
};

} // namespace detail {

} // namespace mcts {
} // namespace multiplayer {
