import 'package:flutter/material.dart';

void main() {
  runApp(const TicTacToeApp());
}

class TicTacToeApp extends StatelessWidget {
  const TicTacToeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tic Tac Toe',
      theme: ThemeData(
        primarySwatch: Colors.blueGrey,
      ),
      home: const GameScreen(),
    );
  }
}

class GameScreen extends StatefulWidget {
  const GameScreen({super.key});

  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> {
  List<String> _board = List.filled(9, '');
  String _currentPlayer = 'X';
  String _winner = '';

  void _makeMove(int index) {
    if (_winner == '' && _board[index] == '') {
      setState(() {
        _board[index] = _currentPlayer;
        _checkWinner();
        if (_winner == '') {
          _currentPlayer = _currentPlayer == 'X' ? 'O' : 'X';
        }
      });
    }
  }

  void _checkWinner() {
    const winningCombinations = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
      [0, 4, 8], [2, 4, 6], // Diagonals
    ];

    for (final combo in winningCombinations) {
      if (_board[combo[0]] != '' &&
          _board[combo[0]] == _board[combo[1]] &&
          _board[combo[0]] == _board[combo[2]]) {
        _winner = _board[combo[0]];
        return;
      }
    }

    if (!_board.any((s) => s == '')) {
      _winner = 'Draw';
    }
  }

  void _resetGame() {
    setState(() {
      _board = List.filled(9, '');
      _currentPlayer = 'X';
      _winner = '';
    });
  }

  Widget _buildSquare(int index) {
    return GestureDetector(
      onTap: () => _makeMove(index),
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey),
        ),
        child: Center(
          child: Text(
            _board[index],
            style: const TextStyle(fontSize: 40),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Tic Tac Toe'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          Expanded(
            child: GridView.builder(
              padding: const EdgeInsets.all(16.0),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
              ),
              itemCount: 9,
              itemBuilder: (context, index) => _buildSquare(index),
            ),
          ),
          if (_winner != '')
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Text(
                _winner == 'Draw' ? 'It\'s a Draw!' : '$_winner wins!',
                style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
            ),
          ElevatedButton(
            onPressed: _resetGame,
            child: const Text('Reset Game'),
          ),
          const SizedBox(height: 30),
        ],
      ),
    );
  }
}