#include "tic_tac_toe.h"

#include <iostream>
#include <algorithm>

using namespace std;

namespace ttt
{   

const array< Move, 3 > wins[8] =
{ { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 }, // rows
  { 0, 3, 6 }, { 1, 4, 7 }, { 2, 5, 8 }, // columns
  { 0, 4, 8 }, { 2, 4, 6 } }; // diagonals


namespace minimax {
double score( State const& state )
{
    double score = 0.0;
    for (auto const& win : wins)
    {
        uint8_t player1_points = 0;
        uint8_t player2_points = 0;
        for (auto const& index : win)
        {
            if (state[index] == Symbol::Player1)
                ++player1_points;
            else if (state[index] == Symbol::Player2)
                ++player2_points;
        }
        if (player1_points == 0)
            score += player2_points;
        else if (player2_points == 0)
            score -= player1_points;
    }

    return score;
}

} // namespace minimax {

namespace alphazero {

float Data::predict( Game const& game, array< float, P >& )
{
    return 0.0;
}

size_t Data::move_to_policy_index( Move const& move ) const
{
    return size_t( move );
}

void Data::serialize_game( 
    Game const& game,
    array< float, G >& game_state_player1,
    array< float, G >& game_state_player2 ) const
{

}

namespace training {

} // namespace training {
} // namespace alphazero {

Symbol player_index_to_symbol( PlayerIndex player_index )
{
    const Symbol symbols[] = { Symbol::Player1, Symbol::Player2 };
    return symbols[player_index];
}

namespace console
{

Move HumanPlayer::choose_move()
{
    vector< Move > valid_moves;
    for (auto itr = game.begin(), end = game.end(); itr != end; ++itr)
        valid_moves.push_back( *itr );

    cout << "valid moves:\n";
    for (auto const& move : valid_moves)
        cout << "(" << int( move ) << "), ";
    cout << '\n';

    while (true)
    {
        cout << "\nmove? ";
        unsigned move;
        cin >> move;
        if (move > 8)
        {
            cout << "invalid move" << endl;
            continue;
        }

        if (!ranges::contains( valid_moves.begin(), valid_moves.end(), Move( move )))
        {
            cout << "not a valid move" << endl;
            continue;
        }

        game = game.apply( Move( move ) );
        return move;
    }
}

void HumanPlayer::apply_opponent_move( Move const& move )
{
    game = game.apply( move );
}

} // namespace console
} // namespace ttt

void GameState< ttt::Move, ttt::State >::next_valid_move( 
    optional< ttt::Move >& move, PlayerIndex, ttt::State const& state )
{
    if (!move)
        move = 0; // first possibly valid move
    else
        ++*move; // next possible move

    while (true)
    {
        if (move >= 9) // no valid move possible anymore
        {
            move.reset();
            break;
        }
        else if (state[*move] == ttt::Symbol::Empty) // move is valid
            break;

        // try next move
        ++*move;
    }
}

void GameState< ttt::Move, ttt::State >::get_valid_moves(
    std::vector< ttt::Move >& moves, PlayerIndex, ttt::State const& state )
{
    // allocate enough (only the first time actually)
    moves.resize( 9 );
    auto move_itr = moves.begin();
    for (char index = 0; index != 9; ++index)
        if (state[index] == ttt::Symbol::Empty)
            *move_itr++ = index;
    // reduce to correct logical size
    moves.resize( move_itr - moves.begin());
}

ttt::State GameState< ttt::Move, ttt::State >::apply( 
    ttt::Move const& move, PlayerIndex player_index, ttt::State const& state )
{
    if (move >= 9)
        throw std::invalid_argument( "invalid move" );
    if (state[move] != ttt::Symbol::Empty)
        throw std::invalid_argument( "invalid move, cell not empty" );

    ttt::State new_state = state;
    new_state[move] = ttt::player_index_to_symbol( player_index );
    return new_state;
}

GameResult GameState< ttt::Move, ttt::State >::result( 
    PlayerIndex player_index, ttt::State const& state )
{
    // check for wins
    for (const auto& win : ttt::wins)
    {
        const ttt::Symbol symbol0 = state[win[0]];
        if (symbol0 != ttt::Symbol::Empty && symbol0 == state[win[1]] && symbol0 == state[win[2]])
            return (symbol0 == ttt::Symbol::Player1) ? GameResult::Player1Win
                                                     : GameResult::Player2Win;
    }

    // check for undecided
    if (std::any_of(state.begin(), state.end(), 
        [](ttt::Symbol symbol) { return symbol == ttt::Symbol::Empty; }))
        return GameResult::Undecided;

    // otherwise its a draw
    return GameResult::Draw;
}    

ostream& operator<<( ostream& os, ttt::Game const& game )
{
    os << "player " << game.current_player_index() + 1 << '\n'
       << "state: " << '\n';
    for (auto i = 0; i != 5; ++i)
        os << '-';
    os << '\n';
    for (size_t index = 0; index != 9; ++index)
    {
        if (index % 3 == 0)
            os << '|';
        os << game.get_state()[index];
        if ((index + 1) % 3 == 0)
            os << "|\n";
    }
    for (auto i = 0; i != 5; ++i)
        os << '-';
    os << '\n';

    return os;
}