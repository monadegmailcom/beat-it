#include "tic_tac_toe.h"

#include <iostream>

using namespace std;

namespace tic_tac_toe
{   

double minimax::Player::score( Game const& game ) const 
{
    const uint8_t scores[9] = 
        { 3, 2, 3, 
          2, 4, 2, 
          3, 2, 3 };

    double score = 0.0;
    State const& state = game.get_state();
    for (size_t i = 0; i != 9; ++i)
    {
        const Symbol s = state[i];
        if (s == Symbol::Empty)
            continue;
        else if (s == Symbol::Player1)
            score -= scores[i];
        else
            score += scores[i];
    }
    return score;
}


GameResult symbol_to_winner( Symbol symbol )
{
    if (symbol == Symbol::Player1)
        return GameResult::Player1Win;
    else if (symbol == Symbol::Player2)
        return GameResult::Player2Win;
    else
        throw std::invalid_argument( "no winner" );

    return GameResult::Draw;
}

Symbol player_index_to_symbol( PlayerIndex player_index )
{
    static const Symbol symbols[] = { Symbol::Player1, Symbol::Player2 };
    return symbols[player_index];
}

Move console::HumanPlayer::choose( Game const& game )
{
    cout << game << endl;

    vector< Move > move_stack;
    game.append_valid_moves( move_stack );

    cout << "\nchoose move from:\n";
    for (size_t index = 0; index != 9; ++index)
    {
        if (game.get_state()[index] == Symbol::Empty)
            cout << index;
        else
            cout << '*';
        if ((index + 1) % 3 == 0)
            cout << '\n';
    }
    cout << endl;

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

        if (find( move_stack.begin(), move_stack.end(), Move( move )) == move_stack.end())
        {
            cout << "not a valid move" << endl;
            continue;
        }

        return move;
    }
}

} // namespace tic_tac_toe

ostream& operator<<( ostream& os, tic_tac_toe::Game const& game )
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