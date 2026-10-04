// Test driver for unmodified, pinned Lc0 sources; see results/upstream-sources.json.
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "neural/encoder.h"

int main() {
  using namespace lczero;
  InitializeMagicBitboards();
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      std::istringstream in(line);
      std::string format, fen, moves;
      std::getline(in, format, '\t');
      std::getline(in, fen, '\t');
      std::getline(in, moves);
      PositionHistory history;
      ChessBoard board;
      int rule50, fullmoves;
      board.SetFromFen(fen, &rule50, &fullmoves);
      history.Reset(board, rule50, (fullmoves - 1) * 2 + board.flipped());
      std::istringstream move_stream(moves);
      std::string uci;
      while (move_stream >> uci) {
        history.Append(Move(uci, history.IsBlackToMove()));
      }
      int transform = -1;
      auto planes = EncodePositionForNN(
          static_cast<pblczero::NetworkFormat::InputFormat>(std::stoi(format)),
          history, 8, FillEmptyHistory::NO, &transform);
      std::cout << "{\"transform\":" << transform << ",\"planes\":[";
      for (int i = 0; i < static_cast<int>(planes.size()); ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << planes[i].mask << ',' << std::setprecision(9)
                  << planes[i].value << ']';
      }
      std::cout << "],\"legal\":[";
      bool first = true;
      for (auto move : history.Last().GetBoard().GenerateLegalMoves()) {
        if (!first) std::cout << ',';
        first = false;
        int index = move.as_nn_index(transform);
        if (history.IsBlackToMove()) move.Mirror();
        std::cout << '[' << index << ",\"" << move.as_string() << "\"]";
      }
      std::cout << "]}\n";
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
