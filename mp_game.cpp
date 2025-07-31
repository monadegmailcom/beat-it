#include "mp_game.h"

using namespace std;

namespace multiplayer
{

ostream& operator<<( ostream& stream, GameResult game_result )
{
    auto visitor = [&stream]( auto&& arg )
    {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (is_same_v< T, Draw >)
            stream << "Draw";
        else if constexpr (is_same_v< T, Undecided >)
            stream << "Undecided";
        else
        {
            auto [ v ] = arg;
            stream << "Winner player " << v;
        }
    };

    visit( visitor, game_result);

    return stream;
}

} // namespace multiplayer