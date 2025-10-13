# Chess AI Game Opponent
## Project Overview
This project is a web-based chess game with an AI opponent and user analytics. It combines a modern JavaScript frontend with a Python Flask backend to deliver an interactive chess experience, player profiling, and Elo prediction.

## Features
- **Play Chess in Browser:** Interactive chessboard UI with move history, captured pieces, and new game/reset options.
- **AI Opponent:** Backend uses Stockfish engine (if available) or a custom Adaptive AI (minimax with profiling) to play against the user.
- **Player Profiling:** The AI adapts its strategy based on the user's playing style (aggressive, defensive, balanced).
- **Elo Prediction:** After each game, the backend can predict the user's approximate Elo rating using a machine learning model (Linear Regression) trained on user game data.
- **Game Data Storage:** User game results (result, blunders, centipawn loss, moves) are stored in a CSV file for analytics and model training.

## Technologies Used
### Frontend
- HTML, CSS, JavaScript (`index.html`, `style.css`, `script.js`)
- Dynamic chessboard rendering and move handling

### Backend
- Python Flask (`backend/app.py`)
- `python-chess` for chess logic and Stockfish integration
- `scikit-learn` for Elo prediction (Linear Regression)
- `pandas`, `numpy` for data handling
- `flask-cors` for cross-origin requests

### Data
- `user_game_data.csv`: Stores user game results for analytics and ML

## AI/ML Techniques Used

### 1. Chess Engine AI
- **Stockfish Integration:**
  - The backend uses the Stockfish chess engine (if available) for move generation. Stockfish is a world-class open-source chess engine that evaluates positions and suggests the best moves using advanced search and evaluation algorithms.
  - If Stockfish is not available, the backend falls back to a custom AI.

### 2. AdaptiveAI (Custom Minimax AI)
- **Minimax Algorithm:**
  - The custom AdaptiveAI uses the minimax algorithm with alpha-beta pruning to search for the best move up to a certain depth.
  - The search depth is dynamically adjusted based on the detected player profile (aggressive, defensive, balanced).
- **Player Profiling:**
  - The AI tracks your moves (captures, checks, pawn pushes, center control) to classify your playing style as Aggressive, Defensive, or Balanced.
  - The AI adapts its move selection strategy based on your profile, making the game more challenging and personalized.

### 2a. Adaptive Difficulty (Dynamic AI Strength)
- **How it works:**
  - The AI automatically adjusts its difficulty based on your recent performance (average Elo from your last 5 games).
  - If you play well (high Elo), the AI increases its search depth and plays more accurately (less randomness).
  - If you struggle (low Elo), the AI reduces its depth and plays with more randomness, making it easier.
  - This ensures a personalized and challenging experience for all skill levels.

### 3. Machine Learning for Elo Prediction
- **Linear Regression Model:**
  - After each game, the backend saves your game stats (result, blunders, centipawn loss, moves) to `user_game_data.csv`.
  - A Linear Regression model (using scikit-learn) is trained on all saved games to learn the relationship between your stats and an estimated Elo rating.
  - The model uses the following features:
    - Number of blunders
    - Average centipawn loss (CPL)
    - Number of moves
    - Game result (mapped to a base Elo)
  - The more games you play, the more accurate the Elo prediction becomes, as the model continuously retrains on your growing dataset.

### 4. Data Handling
- **Pandas & Numpy:**
  - All game data is managed using pandas DataFrames and numpy arrays for efficient storage, retrieval, and model training.
- **CSV Storage:**
  - All user games are stored locally in `user_game_data.csv`, which acts as the training dataset for the ML model.

### 5. Move Quality Feedback (AI/ML)
- **Feature Overview:**
  - After every user move, the backend evaluates the move quality using the Stockfish chess engine.
  - The backend compares your move to the best move in the position and assigns a label: `Best`, `Good`, `Inaccuracy`, `Mistake`, or `Blunder`.
  - This feedback is printed directly in the backend terminal (not shown in the browser), making it ideal for teacher demos or analysis.

- **How it works (Deep Explanation):**
  - When you make a move, the frontend sends the current board position (FEN) and your move (UCI format) to the backend `/move-feedback` endpoint.
  - The backend uses Stockfish to:
    1. Evaluate the position before your move (`eval_before`).
    2. Evaluate the position after your move (`eval_after`).
    3. Find the best move and its evaluation (`best_move`, `best_eval`).
    4. Calculate the difference between your move and the best move.
    5. Assign a label based on the difference:
       - `Best`: Your move matches the engine's top choice.
       - `Good`: Small difference from the best move.
       - `Inaccuracy`, `Mistake`, `Blunder`: Increasingly larger mistakes.
  - All this information is printed in the backend terminal for every user move.

- **Manual Testing Command:**
  - You can manually test this feature from PowerShell or terminal using:
    
    ```powershell
    Invoke-RestMethod -Uri "http://127.0.0.1:5000/move-feedback" -Method Post -Body '{"fen":"<FEN_STRING>","move":"<UCI_MOVE>"}' -ContentType "application/json"
    ```
   Invoke-RestMethod -Uri "http://127.0.0.1:5000/move-feedback" -Method Post -Body '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","move":"e2e4"}' -ContentType "application/json"
    
    Replace `<FEN_STRING>` with the board position and `<UCI_MOVE>` with your move (e.g., `e2e4`).
    
    Example:
    ```powershell
    Invoke-RestMethod -Uri "http://127.0.0.1:5000/move-feedback" -Method Post -Body '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","move":"e2e4"}' -ContentType "application/json"
    ```
    
    The backend terminal will print detailed move feedback for the given move.

### 6. Game Explanation (Why This Move?)
- **Feature Overview:**
  - The project includes a 'Game Explanation' feature where the AI can explain why a particular move was chosen.
  - This helps users understand the reasoning behind AI moves, making the game educational and interactive.

- **How it works:**
  - When the AI (Stockfish or AdaptiveAI) selects a move, it can provide a brief explanation for its choice.
  - For Stockfish, the explanation is based on engine evaluation: the move chosen maximizes the evaluation score, improves position, or avoids threats.
  - For AdaptiveAI, the explanation may include:
    - Capturing a valuable piece (material gain)
    - Delivering check or checkmate
    - Defending a threatened piece
    - Controlling the center or key squares
    - Advancing a passed pawn
    - Avoiding blunders or tactical threats
  - These explanations can be printed in the backend terminal or shown in the UI (if enabled), helping users learn chess strategy and tactics.

- **Example Explanation Output:**
  - "AI played Qd4 to attack your knight and control the center."
  - "AI played e5 to open lines for its bishop and increase pressure."
  - "AI played Rxf7+ to deliver a tactical check and win material."

- **Purpose:**
  - This feature is designed to make the chess experience more transparent and educational, especially for beginners who want to learn not just what to play, but why to play it.

### 8. In-Game Outcome Prediction (Who is Likely to Win?)
- **Feature Overview:**
  - During the game, the backend uses the chess engine's evaluation to predict which side (White or Black) is currently more likely to win.
  - The evaluation score is analyzed after each move, and a simple message like "White likely to win" or "Black likely to win" can be shown based on the position.

- **How it works:**
  - After every move, the engine evaluates the board position.
  - If the evaluation score is strongly positive, it means White is ahead; if strongly negative, Black is ahead.
  - Thresholds are set (e.g., +200 means White is likely to win, -200 means Black is likely to win, otherwise the game is balanced).

- **Purpose:**
  - This gives players real-time insight into who is favored in the current position, helping them understand the impact of each move and learn from the game's flow.

## How to Run
1. Install backend dependencies: `pip install -r backend/requirements.txt`
2. (Optional) Download Stockfish engine and update its path in `app.py`.
3. Run backend: `python backend/app.py`
4. Open `index.html` in your browser to play.

## Folder Structure
- `index.html`, `script.js`, `style.css`, `chess.js`, `three-chess.html`: Frontend files
- `backend/`: Python backend, requirements, and user data
- `pieces/`: (Empty/for future use)

## Notes
- The project can be extended with more advanced AI/ML features (deep learning, puzzle generation, etc.).
- All user data is stored locally in CSV format.

---
Feel free to ask for more details or contribute new features!
