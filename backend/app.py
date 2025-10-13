
# --- Place this route after app = Flask(__name__) ---

import os
import random
import time
import pandas as pd
import chess
import chess.engine
import atexit
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque
import numpy as np

# Ensure better randomization
random.seed(int(time.time()))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"]}}, supports_credentials=True)

# --- Move Quality Feedback Endpoint ---
@app.route('/move-feedback', methods=['POST', 'OPTIONS'])
def move_feedback():
    print("[DEBUG] /move-feedback endpoint hit")
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    print(f"[DEBUG] Incoming data: {data}")
    fen = data.get('fen')
    move_uci = data.get('move')
    if not fen or not move_uci:
        print("[DEBUG] Missing FEN or move in request data")
        return jsonify({'error': 'FEN and move are required'}), 400
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({'error': 'Illegal move'}), 400
        # Evaluate before move
        if stockfish_engine:
            eval_before = stockfish_engine.analyse(board, chess.engine.Limit(time=0.2))['score'].white().score(mate_score=10000)
        else:
            eval_before = 0
        # Evaluate after move
        board.push(move)
        if stockfish_engine:
            eval_after = stockfish_engine.analyse(board, chess.engine.Limit(time=0.2))['score'].white().score(mate_score=10000)
        else:
            eval_after = 0
        # Find best move
        board.pop()
        best_move = None
        best_eval = eval_before
        if stockfish_engine:
            best = stockfish_engine.analyse(board, chess.engine.Limit(time=0.2), multipv=3)
            if isinstance(best, list):
                best_move = best[0]['pv'][0].uci() if 'pv' in best[0] and best[0]['pv'] else None
                best_eval = best[0]['score'].white().score(mate_score=10000)
            elif isinstance(best, dict) and 'pv' in best and best['pv']:
                best_move = best['pv'][0].uci()
                best_eval = best['score'].white().score(mate_score=10000)
        # Calculate difference
        diff = eval_after - best_eval
        # Label
        if abs(diff) < 30:
            label = 'Best'
        elif abs(diff) < 100:
            label = 'Good'
        elif abs(diff) < 200:
            label = 'Inaccuracy'
        elif abs(diff) < 400:
            label = 'Mistake'
        else:
            label = 'Blunder'
        # Print feedback in terminal (for teacher demo)
        print("[MOVE FEEDBACK]")
        print(f"best_eval   : {best_eval}")
        print(f"best_move   : {best_move}")
        print(f"difference  : {diff}")
        print(f"eval_after  : {eval_after}")
        print(f"eval_before : {eval_before}")
        print(f"label       : {label}")
        print(f"your_move   : {move_uci}")
        return jsonify({
            'your_move': move_uci,
            'best_move': best_move,
            'eval_before': eval_before,
            'eval_after': eval_after,
            'best_eval': best_eval,
            'difference': diff,
            'label': label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Game Outcome Prediction Model ---
class GameOutcomePredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def extract_position_features(self, board):
        """Extract features from chess position for ML prediction"""
        features = {}
        
        # Material count
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        white_material = black_material = 0
        white_pieces = black_pieces = 0
        
        # Piece mobility and attacks
        white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
        board.turn = not board.turn  # Switch turn to count black mobility
        black_mobility = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
        board.turn = not board.turn  # Switch back
        
        # Count pieces and material
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                    white_pieces += 1
                else:
                    black_material += value
                    black_pieces += 1
        
        # King safety (simplified)
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        white_king_attacks = len(board.attackers(chess.BLACK, white_king_square))
        black_king_attacks = len(board.attackers(chess.WHITE, black_king_square))
        
        features = {
            'material_advantage': white_material - black_material,
            'piece_advantage': white_pieces - black_pieces,
            'mobility_advantage': white_mobility - black_mobility,
            'king_safety_advantage': black_king_attacks - white_king_attacks,
            'game_phase': min(white_pieces + black_pieces, 32) / 32.0,  # 0=endgame, 1=opening
            'white_material': white_material,
            'black_material': black_material,
            'total_moves': len(board.move_stack)
        }
        
        return list(features.values())
    
    def predict_outcome(self, board):
        """Predict game outcome: 0=Black wins, 1=Draw, 2=White wins"""
        try:
            features = self.extract_position_features(board)
            
            # Simple heuristic if model not trained
            if not self.is_trained:
                material_diff = features[0]  # material_advantage
                if material_diff > 3:
                    return 2, 0.8  # White likely wins
                elif material_diff < -3:
                    return 0, 0.8  # Black likely wins
                else:
                    return 1, 0.6  # Draw likely
            
            # Use trained model
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = max(probabilities)
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return 1, 0.5  # Default to draw with low confidence
    
    def get_prediction_text(self, prediction, confidence):
        """Convert prediction to human readable text"""
        outcomes = {
            0: "Black likely to win",
            1: "Position is balanced", 
            2: "White likely to win"
        }
        
        confidence_text = ""
        if confidence > 0.8:
            confidence_text = " (High confidence)"
        elif confidence > 0.6:
            confidence_text = " (Medium confidence)"
        else:
            confidence_text = " (Low confidence)"
            
        return outcomes.get(prediction, "Unclear") + confidence_text

# Initialize game outcome predictor
game_predictor = GameOutcomePredictor()

def train_elo_model(data_path="user_game_data.csv"):
    if not os.path.exists(data_path):
        return None, None
    
    try:
        # Read CSV with headers
        df = pd.read_csv(data_path)
        print(f"[DEBUG] CSV loaded: {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Use elo column if available, otherwise map from result
        if 'elo' in df.columns:
            X = df[["blunders","cpl","moves"]]
            y = df["elo"]
        else:
            result_map = {"1-0": 1400, "0-1": 1000, "1/2-1/2": 1200, "win": 1400, "loss": 1000, "draw": 1200}
            df = df[df["result"].isin(result_map.keys())]
            df["elo"] = df["result"].map(result_map)
            X = df[["blunders","cpl","moves"]]
            y = df["elo"]
        
        if len(X) < 3:
            print(f"[DEBUG] Not enough data: {len(X)} rows")
            return None, None
            
        print(f"[DEBUG] Training model with {len(X)} samples")
        model = LinearRegression()
        model.fit(X, y)
        print(f"[DEBUG] Model trained successfully")
        return model, X.columns
        
    except Exception as e:
        print(f"[ERROR] Failed to train model: {e}")
        return None, None

def predict_elo(blunders, cpl, moves, model, feature_names):
    if model is None or feature_names is None:
        return None
    
    # Create DataFrame with proper feature names to avoid warnings
    import pandas as pd
    X_pred = pd.DataFrame({
        'blunders': [blunders],
        'cpl': [cpl], 
        'moves': [moves]
    })
    
    elo_pred = model.predict(X_pred)[0]
    return float(elo_pred)

class AdaptiveAI:
    def __init__(self, color=chess.BLACK, window=10):
        self.color = color
        self.window = window
        self.moves_window = deque(maxlen=window)
        self.captures = 0
        self.checks = 0
        self.pawn_pushes = 0
        self.total_tracked = 0
        self.center_control = 0
        self.move_count = 0
        self.adaptive_depth = 3
        self.adaptive_randomness = True

    def update_difficulty(self, recent_elo):
        """
        Adjust AI difficulty based on recent Elo (last 5 games average).
        Higher Elo = higher depth, less randomness. Lower Elo = easier AI.
        """
        if recent_elo is None:
            self.adaptive_depth = 3
            self.adaptive_randomness = True
            return
        if recent_elo >= 1350:
            self.adaptive_depth = 5
            self.adaptive_randomness = False
        elif recent_elo >= 1200:
            self.adaptive_depth = 4
            self.adaptive_randomness = False
        elif recent_elo >= 1100:
            self.adaptive_depth = 3
            self.adaptive_randomness = True
        else:
            self.adaptive_depth = 2
            self.adaptive_randomness = True

    def reset(self):
        print(f"[DEBUG] AdaptiveAI.reset() called - move_count before reset: {self.move_count}")
        self.moves_window.clear()
        self.captures = 0
        self.checks = 0
        self.pawn_pushes = 0
        self.total_tracked = 0
        self.center_control = 0
        self.move_count = 0
        print(f"[DEBUG] AdaptiveAI.reset() completed - move_count after reset: {self.move_count}")

    def register_player_move(self, board: chess.Board):
        if not board.move_stack:
            return
        last_move = board.peek()
        is_capture = board.is_capture(last_move)
        gives_check = board.is_check()
        promotion = last_move.promotion is not None
        moved_piece_type = board.piece_type_at(last_move.to_square)
        pawn_push = (moved_piece_type == chess.PAWN)
        center_sqs = {chess.D4, chess.E4, chess.D5, chess.E5}
        if last_move.to_square in center_sqs:
            self.center_control += 1
        if is_capture:
            self.captures += 1
        if gives_check:
            self.checks += 1
        if pawn_push:
            self.pawn_pushes += 1
        if promotion:
            self.pawn_pushes += 1
        self.total_tracked += 1

    def get_player_profile(self):
        if self.total_tracked == 0:
            return "Unknown"
        aggression = (self.captures + self.checks + self.pawn_pushes) / self.total_tracked
        defense = (self.total_tracked - self.captures - self.pawn_pushes) / self.total_tracked
        if aggression > 0.6:
            return "Aggressive"
        elif defense > 0.6:
            return "Defensive"
        else:
            return "Balanced"

    def evaluate_board(self, board: chess.Board):
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Check if either king is missing (game should be over)
        white_king = board.king(chess.WHITE) is not None
        black_king = board.king(chess.BLACK) is not None
        
        # Massive bonus/penalty for king captures
        if not white_king:
            return -999999  # Black wins (very negative for white)
        if not black_king:
            return 999999   # White wins (very positive for white)
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        
        # Add king safety evaluation
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        # Penalty for king being under attack
        white_king_attackers = len(board.attackers(chess.BLACK, white_king_square))
        black_king_attackers = len(board.attackers(chess.WHITE, black_king_square))
        
        score -= white_king_attackers * 50  # Penalty for white king being attacked
        score += black_king_attackers * 50  # Bonus for attacking black king
        
        return score

    def minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board), None
        best_move = None
        if maximizing:
            max_eval = float("-inf")
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_ai_move(self, board: chess.Board, randomize=None, top_n=3, force_random=False):
        # Use adaptive difficulty
        depth = getattr(self, 'adaptive_depth', 3)
        use_random = self.adaptive_randomness if randomize is None else randomize
        
        # Safety check: ensure there are legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print("[ERROR] No legal moves available in get_ai_move!")
            return None
            
        if self.move_count == 0 or force_random:
            return random.choice(legal_moves)
        
        # CRITICAL FIX: Check for immediate king captures first
        king_capture_moves = []
        for move in legal_moves:
            target_piece = board.piece_at(move.to_square)
            if target_piece and target_piece.piece_type == chess.KING:
                king_capture_moves.append(move)
                print(f"[CRITICAL] Found king capture move: {move.uci()}")
        
        # Always prioritize king captures
        if king_capture_moves:
            selected_move = random.choice(king_capture_moves)
            print(f"[CRITICAL] AI selecting king capture: {selected_move.uci()}")
            return selected_move
            
        profile = self.get_player_profile()
        # Optionally, profile can still influence depth
        if profile == "Aggressive":
            depth = max(depth, 4)
        elif profile == "Defensive":
            depth = max(depth, 3)
            
        moves_scores = []
        try:
            for move in board.legal_moves:
                board.push(move)
                score, _ = self.minimax(board, depth - 1, float("-inf"), float("inf"), False)
                board.pop()
                moves_scores.append((move, score))
        except Exception as e:
            print(f"[ERROR] Minimax evaluation failed: {e}")
            return random.choice(legal_moves)  # Fallback to random
            
        moves_scores.sort(key=lambda x: -x[1])
        if use_random and len(moves_scores) > 0:
            top_moves = [m[0] for m in moves_scores[:top_n]]
            return random.choice(top_moves)
        if moves_scores:
            return moves_scores[0][0]
        
        # Final fallback
        return random.choice(legal_moves)

STOCKFISH_PATH = r"C:\\Users\\akash\\Downloads\\stockfish\\stockfish-windows-x86-64-avx2.exe"

def start_stockfish():
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print(f"[INFO] Stockfish engine started at {STOCKFISH_PATH}.")
        return engine
    except Exception as e:
        print("[WARNING] Stockfish not found or failed to start:", e)
        return None

stockfish_engine = start_stockfish()
ai_engine = AdaptiveAI()

@app.route('/ai-move', methods=['POST', 'OPTIONS'])
def ai_move():
    # --- Adaptive Difficulty: Calculate recent Elo and update AI ---
    try:
        df = pd.read_csv("user_game_data.csv", names=["result","blunders","cpl","moves"])
        result_map = {"1-0": 1400, "0-1": 1000, "1/2-1/2": 1200, "win": 1400, "loss": 1000, "draw": 1200}
        df = df[df["result"].isin(result_map.keys())]
        df["elo"] = df["result"].map(result_map)
        recent_elo = df["elo"].tail(5).mean() if not df.empty else None
    except Exception as e:
        print(f"[WARNING] Could not read user_game_data.csv for adaptive difficulty: {e}")
        recent_elo = None
    ai_engine.update_difficulty(recent_elo)
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    fen = data.get('fen')
    last_player_move = data.get('last_move')
    is_new_game = data.get('is_new_game', False)

    if not fen:
        return jsonify({'error': 'FEN not provided'}), 400

    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({'error': f'Invalid FEN: {str(e)}'}), 400

    # Check for no legal moves (checkmate or stalemate) - REMOVED insufficient material check
    if not board.legal_moves or board.is_game_over():
        if board.is_checkmate():
            winner = 'white' if board.turn == chess.BLACK else 'black'
            print(f"[GAME OVER] Checkmate! Winner: {winner}")
            return jsonify({
                'game_over': True,
                'reason': 'checkmate',
                'winner': winner,
                'fen': fen,
                'message': f'Checkmate! {winner.capitalize()} wins!'
            })
        elif board.is_stalemate():
            print("[GAME OVER] Stalemate!")
            return jsonify({
                'game_over': True,
                'reason': 'stalemate',
                'winner': 'draw',
                'fen': fen,
                'message': 'Stalemate! Game is a draw.'
            })
        else:
            print("[GAME OVER] Game over - other condition")
            return jsonify({
                'game_over': True,
                'reason': 'game_over',
                'winner': 'draw',
                'fen': fen,
                'message': 'Game over!'
            })

    # Enhanced game state detection
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    total_moves_in_game = len(board.move_stack)
    is_starting_position = (fen == starting_fen)
    
    # More comprehensive new game detection
    is_very_early_game = total_moves_in_game <= 2  # First couple moves
    is_reset_scenario = (is_new_game or is_starting_position or 
                        (is_very_early_game and ai_engine.move_count > 2))
    
    print(f"[DEBUG] ========== MOVE REQUEST ==========")
    print(f"[DEBUG] FEN: {fen}")
    print(f"[DEBUG] Total moves in game: {total_moves_in_game}")
    print(f"[DEBUG] Is starting position: {is_starting_position}")
    print(f"[DEBUG] AI move count before check: {ai_engine.move_count}")
    print(f"[DEBUG] is_new_game flag: {is_new_game}")
    print(f"[DEBUG] is_very_early_game: {is_very_early_game}")
    print(f"[DEBUG] is_reset_scenario: {is_reset_scenario}")
    
    # CRITICAL FIX: Reset AI state for new games
    if is_reset_scenario:
        ai_engine.reset()
        print(f"[INFO] *** GAME RESET *** - Reason: new_game={is_new_game}, starting_pos={is_starting_position}, early_game={is_very_early_game}")
    else:
        print(f"[INFO] Continuing game (Total moves: {total_moves_in_game}, AI moves: {ai_engine.move_count})")

    # Register player move for profiling
    if last_player_move:
        try:
            board.push_uci(last_player_move)
            ai_engine.register_player_move(board)
            board.pop()
        except Exception as ex:
            print(f"[WARNING] Invalid last_move provided: {ex}")

    TOP_N = 3
    move = None
    engine_used = 'adaptive'
    stockfish_eval = None

    # FIXED: Check if this is first AI move AFTER potential reset
    is_first_ai_move = (ai_engine.move_count == 0)
    
    print(f"[DEBUG] After reset check - AI move_count: {ai_engine.move_count}")
    print(f"[DEBUG] is_first_ai_move: {is_first_ai_move}")
    
    if is_first_ai_move:
        print(f"[DEBUG] *** MAKING FIRST AI MOVE ***")
        
        # For first move, make it completely random from ALL legal moves
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            print("[CRITICAL] No legal moves available for first AI move!")
            return jsonify({'error': 'No legal moves available'}), 500
        
        # Debug: Show what types of moves are available
        move_types = {}
        for move_option in legal_moves:
            piece = board.piece_at(move_option.from_square)
            if piece:
                piece_name = chess.piece_name(piece.piece_type).title()
                if piece_name not in move_types:
                    move_types[piece_name] = []
                move_types[piece_name].append(move_option.uci())
        
        print(f"[DEBUG] Total legal moves available for first move: {len(legal_moves)}")
        print(f"[DEBUG] Available moves by piece type:")
        for piece_type, moves in move_types.items():
            print(f"  {piece_type}: {moves}")
        
        # Add extra randomization with time-based seed
        random.seed(int(time.time() * 1000000) % 1000000)
        
        # Select random move
        move = random.choice(legal_moves)
        ai_engine.move_count += 1  # Increment AFTER selection
        engine_used = 'random_first_move'
        
        print(f"[DEBUG] After first move selection, AI move_count: {ai_engine.move_count}")
        
        # Get Stockfish evaluation for the random move
        if stockfish_engine:
            try:
                result = stockfish_engine.analyse(board, chess.engine.Limit(time=0.1))
                if isinstance(result, dict) and 'score' in result:
                    stockfish_eval = result['score'].white().score(mate_score=10000)
            except Exception as e:
                print("[WARNING] Stockfish evaluation error on random move:", e)
        
        selected_piece = board.piece_at(move.from_square)
        selected_piece_name = chess.piece_name(selected_piece.piece_type).title() if selected_piece else "Unknown"
        print(f"[INFO] AI selected RANDOM first move: {move.uci()} using {selected_piece_name} from {len(legal_moves)} total options")
    
    else:
        print(f"[DEBUG] Making strategic move (not first move)")
        
        # CRITICAL FIX: Check for immediate king captures first
        legal_moves = list(board.legal_moves)
        king_capture_moves = []
        
        for move in legal_moves:
            target_piece = board.piece_at(move.to_square)
            if target_piece and target_piece.piece_type == chess.KING:
                king_capture_moves.append(move)
                print(f"[CRITICAL] Stockfish found king capture move: {move.uci()}")
        
        # Always prioritize king captures over Stockfish analysis
        if king_capture_moves:
            move = random.choice(king_capture_moves)
            engine_used = 'king_capture_priority'
            print(f"[CRITICAL] Prioritizing king capture: {move.uci()}")
            # Get Stockfish evaluation for the king capture move
            if stockfish_engine:
                try:
                    result = stockfish_engine.analyse(board, chess.engine.Limit(time=0.1))
                    if isinstance(result, dict) and 'score' in result:
                        stockfish_eval = result['score'].white().score(mate_score=10000)
                except Exception as e:
                    print("[WARNING] Stockfish evaluation error on king capture:", e)
                    stockfish_eval = 999999  # Assume mate value for king capture
        else:
            # For subsequent moves without king captures, use Stockfish strategically
            if stockfish_engine:
                try:
                    result = stockfish_engine.analyse(board, chess.engine.Limit(time=0.5), multipv=TOP_N)
                    top_moves = []
                    eval_score = None

                    if isinstance(result, list):
                        for r in result:
                            try:
                                if 'pv' in r and r['pv'] and isinstance(r['pv'][0], chess.Move):
                                    top_moves.append(r['pv'][0])
                                if eval_score is None and 'score' in r:
                                    eval_score = r['score'].white().score(mate_score=10000)
                            except Exception:
                                continue
                    elif isinstance(result, dict):
                        if 'pv' in result and result['pv'] and isinstance(result['pv'][0], chess.Move):
                            top_moves.append(result['pv'][0])
                        if 'score' in result:
                            eval_score = result['score'].white().score(mate_score=10000)
                    
                    if top_moves:
                        move = random.choice(top_moves)
                        engine_used = 'stockfish'
                    else:
                        move = None
                    stockfish_eval = eval_score
                except Exception as e:
                    print("[WARNING] Stockfish error:", e)
                    move = None

            # Fallback to adaptive AI if Stockfish fails
            if not move:
                move = ai_engine.get_ai_move(board, randomize=True, top_n=TOP_N)
                engine_used = 'adaptive'
                ai_engine.move_count += 1  # Increment for adaptive AI moves too
                
                # Get evaluation from our adaptive AI when Stockfish unavailable
                if stockfish_eval is None:
                    try:
                        ai_eval = ai_engine.evaluate_board(board)
                        stockfish_eval = ai_eval / 10  # Scale down to centipawn-like values
                        print(f"[INFO] Using adaptive AI evaluation: {stockfish_eval}")
                    except Exception as e:
                        print(f"[WARNING] Adaptive evaluation error: {e}")
                        stockfish_eval = 0  # Neutral evaluation as fallback
            else:
                # Increment move count for Stockfish moves (except first move which was already incremented)
                if not is_first_ai_move:
                    ai_engine.move_count += 1
        
        print(f"[DEBUG] Strategic move selected: {move.uci() if move else 'None'}")

    # Safety check with fallback
    if not isinstance(move, chess.Move):
        print("ERROR: move type is not chess.Move! Received:", type(move), move)
        print("[EMERGENCY] Generating fallback move...")
        
        # Emergency fallback: get ANY legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            engine_used = 'emergency_fallback'
            print(f"[EMERGENCY] Selected fallback move: {move.uci()}")
        else:
            print("[CRITICAL] No legal moves available!")
            return jsonify({'error': 'No legal moves available - game should be over'}), 500

    board.push(move)

    def predict_outcome(eval_score, board=None):
        # If Stockfish evaluation is available
        if eval_score is not None:
            if eval_score > 200:
                return "White likely to win"
            elif eval_score < -200:
                return "Black likely to win"
            else:
                return "Unclear/Equal"
        
        # Fallback: Use our own board evaluation when Stockfish fails
        if board:
            try:
                ai_eval = ai_engine.evaluate_board(board)
                
                if ai_eval > 500:
                    return "White likely to win"
                elif ai_eval < -500:
                    return "Black likely to win"
                elif abs(ai_eval) < 100:
                    return "Equal position"
                else:
                    return "Slight advantage"
            except Exception as e:
                print(f"[WARNING] Board evaluation error: {e}")
        
        if board and len(board.move_stack) > 40:
            return "Endgame - outcome unclear"
        elif board and len(board.move_stack) < 10:
            return "Opening - early to predict"
        else:
            return "Mid-game position"

    outcome = predict_outcome(stockfish_eval, board)

    # Natural language explanation for the AI move
    def explain_move(move, board_before, board_after):
        piece = board_before.piece_at(move.from_square)
        piece_name = chess.piece_name(piece.piece_type).title() if piece else "Piece"
        explanation = f"{piece_name} moved from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}."
        # Check for capture
        captured = board_after.is_capture(move)
        if captured:
            captured_piece = board_before.piece_at(move.to_square)
            if captured_piece:
                explanation += f" Captured {chess.piece_name(captured_piece.piece_type)}."
        # Check for check
        if board_after.is_check():
            explanation += " This move gives check."
        # Center control
        if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
            explanation += " Controls the center."
        # Promotion
        if move.promotion:
            explanation += f" Promoted to {chess.piece_name(move.promotion)}."
        # Default
        return explanation

    # Make a copy of the board before move for explanation
    import copy
    board_before = copy.deepcopy(board)
    # board already has move pushed, so pop and push for before/after
    board.pop()
    board_after = copy.deepcopy(board)
    board.push(move)
    explanation = explain_move(move, board_before, board)

    # Get outcome prediction for current position
    try:
        outcome_pred, outcome_conf = game_predictor.predict_outcome(board)
        outcome_text = game_predictor.get_prediction_text(outcome_pred, outcome_conf)
    except Exception as e:
        print(f"[WARNING] Outcome prediction failed: {e}")
        outcome_pred, outcome_conf, outcome_text = 1, 0.5, "Position unclear"

    response = {
        'move': move.uci(),
        'fen': board.fen(),
        'profile': ai_engine.get_player_profile(),
        'engine': engine_used,
        'outcome_prediction': outcome_text,
        'outcome_confidence': round(outcome_conf, 3),
        'eval_score': stockfish_eval if stockfish_eval is not None else "N/A",
        'move_number': ai_engine.move_count,
        'is_first_move': is_first_ai_move,
        'game_state': 'new_game' if is_first_ai_move else 'ongoing',
        'explanation': explanation,
        'position_analysis': {
            'material_balance': 'Check /predict-outcome for detailed analysis',
            'recommended_strategy': 'Focus on ' + ('attack' if outcome_pred == 2 else 'defense' if outcome_pred == 0 else 'balanced play')
        }
    }
    
    print(f"[DEBUG] ========== FINAL RESPONSE ==========")
    print("AI Response:", response)
    print(f"[DEBUG] ====================================")
    return jsonify(response)

# Add a manual reset endpoint for debugging
@app.route('/force-reset', methods=['POST', 'OPTIONS'])
def force_reset():
    if request.method == 'OPTIONS':
        return '', 200
    
    global ai_engine
    ai_engine = AdaptiveAI()  # Fresh instance
    print("[INFO] *** FORCED RESET *** - AI completely reset")
    return jsonify({
        'message': 'AI forcefully reset', 
        'ai_move_count': ai_engine.move_count,
        'status': 'fresh_start'
    })

@app.route('/predict-outcome', methods=['POST', 'OPTIONS'])
def predict_game_outcome():
    """Real-time game outcome prediction endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    fen = data.get('fen')
    
    if not fen:
        return jsonify({'error': 'FEN string required'}), 400
    
    try:
        board = chess.Board(fen)
        
        # Get prediction
        prediction, confidence = game_predictor.predict_outcome(board)
        prediction_text = game_predictor.get_prediction_text(prediction, confidence)
        
        # Extract additional analysis
        features = game_predictor.extract_position_features(board)
        
        response = {
            'prediction': int(prediction),
            'confidence': round(confidence, 3),
            'prediction_text': prediction_text,
            'analysis': {
                'material_advantage': features[0],
                'mobility_advantage': features[2], 
                'game_phase': round(features[4], 2),
                'total_moves': features[7]
            },
            'recommendations': get_position_recommendations(board, prediction)
        }
        
        print(f"[OUTCOME PREDICTION] {prediction_text} (Confidence: {confidence:.2f})")
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] Outcome prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

def get_position_recommendations(board, prediction):
    """Get strategic recommendations based on position"""
    recommendations = []
    
    # Basic recommendations based on prediction
    if prediction == 2:  # White advantage
        recommendations.append("White should maintain pressure and avoid trades")
        recommendations.append("Look for tactical opportunities to increase advantage")
    elif prediction == 0:  # Black advantage  
        recommendations.append("Black should consolidate advantage")
        recommendations.append("Consider simplifying to winning endgame")
    else:  # Balanced
        recommendations.append("Position is roughly equal")
        recommendations.append("Look for imbalances to create winning chances")
    
    # Game phase recommendations
    if len(board.move_stack) < 20:
        recommendations.append("Focus on piece development and king safety")
    elif len(board.move_stack) > 60:
        recommendations.append("Endgame: activate king and push passed pawns")
    else:
        recommendations.append("Middlegame: look for tactical combinations")
    
    return recommendations

# Add endpoint to check current AI state
@app.route('/ai-status', methods=['GET'])
def ai_status():
    return jsonify({
        'move_count': ai_engine.move_count,
        'profile': ai_engine.get_player_profile(),
        'total_tracked': ai_engine.total_tracked,
        'is_fresh_start': ai_engine.move_count == 0
    })

# --- ELO Prediction Endpoint ---
@app.route('/predict-elo', methods=['POST', 'OPTIONS'])
def predict_elo_api():
    if request.method == 'OPTIONS':
        return '', 200
    
    print("[DEBUG] predict-elo endpoint called")
    data = request.get_json()
    print(f"[DEBUG] Received data: {data}")
    
    # Validate input
    required_fields = ["blunders", "cpl", "moves"]
    if not all(field in data for field in required_fields):
        print(f"[ERROR] Missing fields. Got: {list(data.keys())}")
        return jsonify({"error": "Missing required fields: blunders, cpl, moves"}), 400
    
    try:
        blunders = int(data["blunders"])
        cpl = float(data["cpl"])
        moves = int(data["moves"])
        print(f"[DEBUG] Parsed values - blunders: {blunders}, cpl: {cpl}, moves: {moves}")
    except Exception as e:
        print(f"[ERROR] Invalid input types: {e}")
        return jsonify({"error": f"Invalid input types: {e}"}), 400

    # Train/load model
    print("[DEBUG] Training/loading model...")
    model, feature_names = train_elo_model("user_game_data.csv")
    if model is None or feature_names is None:
        print("[ERROR] Model training failed")
        return jsonify({"error": "Not enough data to train Elo model. Play and save more games first."}), 400

    # Predict Elo
    try:
        print("[DEBUG] Making prediction...")
        elo = predict_elo(blunders, cpl, moves, model, feature_names)
        print(f"[DEBUG] Predicted Elo: {elo}")
        return jsonify({"predicted_elo": round(elo, 2)})
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route('/save-game-data', methods=['POST', 'OPTIONS'])
def save_game_data():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        # Extract game data
        result = data.get('result', 'draw')
        blunders = data.get('blunders', 0)
        cpl = data.get('cpl', 0)
        moves = data.get('moves', 1)
        
        # Estimate Elo based on performance
        if result == 'win':
            estimated_elo = 1200 + (50 - blunders * 10) - (cpl * 2)
        elif result == 'loss':
            estimated_elo = 1000 + (30 - blunders * 8) - (cpl * 1.5)
        else:  # draw
            estimated_elo = 1150 + (40 - blunders * 9) - (cpl * 1.8)
        
        estimated_elo = max(800, min(1600, estimated_elo))  # Clamp between 800-1600
        
        # Save to CSV
        import csv
        with open('user_game_data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([result, blunders, cpl, moves, round(estimated_elo)])
        
        return jsonify({
            'success': True, 
            'estimated_elo': round(estimated_elo),
            'message': 'Game data saved successfully'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to save game data: {e}'}), 500

@app.route('/analyze-game', methods=['POST', 'OPTIONS'])
def analyze_game():
    """Analyze game and provide feedback on better moves after loss"""
    if request.method == 'OPTIONS':
        return '', 200
    
    print("\n" + "="*60)
    print("🎯 POST-GAME ANALYSIS - FEEDBACK ON BETTER MOVES")
    print("="*60)
    
    data = request.get_json()
    move_history = data.get('move_history', [])
    final_fen = data.get('final_fen')
    
    print(f"📋 Total moves played: {len(move_history)}")
    print(f"📍 Final position: {final_fen}")
    
    if not move_history or not final_fen:
        return jsonify({'error': 'Move history and final FEN required'}), 400
    
    # Check if Stockfish engine is alive
    global stockfish_engine
    if not stockfish_engine:
        print("[WARNING] Stockfish engine not available - restarting...")
        stockfish_engine = start_stockfish()
        if not stockfish_engine:
            print("[ERROR] Could not restart Stockfish engine")
            return jsonify({
                'success': False,
                'error': 'Analysis engine unavailable',
                'feedback': [],
                'message': 'Engine offline - try restarting the Python server'
            })
    
    try:
        feedback_moves = []
        
        # Analyze last 5 critical moves
        positions_to_analyze = min(5, len(move_history) // 2)
        print(f"\n🔍 Analyzing last {positions_to_analyze} critical positions...\n")
        
        board = chess.Board()
        
        # Validate and replay moves to get positions
        all_positions = [board.fen()]
        valid_moves = []
        
        for i, move_uci in enumerate(move_history):
            try:
                # Validate move before pushing
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    all_positions.append(board.fen())
                    valid_moves.append(move_uci)
                else:
                    print(f"[WARNING] Illegal move {move_uci} at position {i} - skipping")
                    continue
            except Exception as e:
                print(f"[WARNING] Invalid move format '{move_uci}' at position {i}: {e} - skipping")
                continue
        
        print(f"[INFO] Validated {len(valid_moves)} out of {len(move_history)} moves")
        
        # Analyze last few white moves (user moves) using valid moves only
        for i in range(len(valid_moves) - positions_to_analyze * 2, len(valid_moves), 2):
            if i < 0 or i >= len(valid_moves):
                continue
                
            try:
                # Reconstruct board at this position using valid moves
                temp_board = chess.Board()
                for j in range(i):
                    if j < len(valid_moves):
                        try:
                            temp_board.push_uci(valid_moves[j])
                        except:
                            continue
                
                # Get the move that was played
                if i >= len(valid_moves):
                    continue
                    
                played_move = valid_moves[i]
                move_number = (i // 2) + 1
                
                print(f"\n{'─'*50}")
                print(f"📍 MOVE {move_number} ANALYSIS")
                print(f"{'─'*50}")
                
                # Test engine before analysis
                try:
                    test_result = stockfish_engine.analyse(temp_board, chess.engine.Limit(time=0.1))
                    if not test_result:
                        print(f"[ERROR] Engine test failed - restarting Stockfish")
                        stockfish_engine.quit()
                        stockfish_engine = start_stockfish()
                        if not stockfish_engine:
                            raise Exception("Could not restart engine")
                except Exception as engine_error:
                    print(f"[ERROR] Engine failure: {engine_error} - attempting restart")
                    try:
                        if stockfish_engine:
                            stockfish_engine.quit()
                    except:
                        pass
                    stockfish_engine = start_stockfish()
                    if not stockfish_engine:
                        print(f"[ERROR] Engine restart failed for move {move_number}")
                        continue
                
                # Analyze with Stockfish
                if stockfish_engine:
                    result = stockfish_engine.analyse(temp_board, chess.engine.Limit(time=0.5), multipv=3)
                    
                    alternatives = []
                    if isinstance(result, list):
                        for r in result[:3]:
                            if 'pv' in r and r['pv']:
                                move = r['pv'][0]
                                eval_score = r['score'].white().score(mate_score=10000) if 'score' in r else 0
                                alternatives.append({
                                    'move': move.uci(),
                                    'san': temp_board.san(move),
                                    'evaluation': eval_score
                                })
                    
                    # Get evaluation of played move
                    try:
                        temp_board.push_uci(played_move)
                        played_eval = stockfish_engine.analyse(temp_board, chess.engine.Limit(time=0.3))['score'].white().score(mate_score=10000)
                        temp_board.pop()
                    except Exception as eval_error:
                        print(f"[ERROR] Could not evaluate played move {played_move}: {eval_error}")
                        continue
                    
                    # Print detailed feedback to terminal
                    print(f"\n❌ You played: {played_move}")
                    print(f"   Evaluation: {played_eval:+.1f}")
                    
                    if alternatives:
                        print(f"\n✅ Better alternatives:")
                        for idx, alt in enumerate(alternatives, 1):
                            improvement = alt['evaluation'] - played_eval
                            improvement_symbol = "📈" if improvement > 0 else "📉"
                            print(f"   {idx}. {alt['san']:8} ({alt['move']:8}) - Eval: {alt['evaluation']:+7.1f} {improvement_symbol} ({improvement:+.1f})")
                    
                    feedback_moves.append({
                        'move_number': move_number,
                        'your_move': played_move,
                        'your_eval': played_eval,
                        'alternatives': alternatives
                    })
            except Exception as e:
                print(f"[ERROR] Analysis failed for move {i}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ Analysis complete! Found {len(feedback_moves)} positions to improve")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'feedback': feedback_moves
        })
        
    except Exception as e:
        print(f"[ERROR] Game analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

@atexit.register
def cleanup():
    if stockfish_engine:
        stockfish_engine.quit()

if __name__ == '__main__':
    app.run(debug=True)