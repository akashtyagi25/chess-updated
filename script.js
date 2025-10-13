// --- Chess Endgame Logic: Checkmate & Stalemate ---
function isKingInCheck(color) {
    // Find king position
    let kingRow = -1, kingCol = -1;
    const kingSymbol = color === 'white' ? '♔' : '♚';
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            if (gameBoard[r][c] === kingSymbol) {
                kingRow = r; kingCol = c;
            }
        }
    }
    if (kingRow === -1) return false; // King not on board
    // Is king attacked?
    const attackerColor = color === 'white' ? 'black' : 'white';
    return isSquareAttacked(kingRow, kingCol, attackerColor);
}

function hasAnyLegalMove(color) {
    for (let fromRow = 0; fromRow < 8; fromRow++) {
        for (let fromCol = 0; fromCol < 8; fromCol++) {
            const piece = gameBoard[fromRow][fromCol];
            if (!piece) continue;
            if (color === 'white' && !Object.values(pieces.white).includes(piece)) continue;
            if (color === 'black' && !Object.values(pieces.black).includes(piece)) continue;
            for (let toRow = 0; toRow < 8; toRow++) {
                for (let toCol = 0; toCol < 8; toCol++) {
                    if (fromRow === toRow && fromCol === toCol) continue;
                    if (!isValidMove(fromRow, fromCol, toRow, toCol)) continue;
                    // Try move, check if king is safe
                    const backupFrom = gameBoard[fromRow][fromCol];
                    const backupTo = gameBoard[toRow][toCol];
                    gameBoard[toRow][toCol] = gameBoard[fromRow][fromCol];
                    gameBoard[fromRow][fromCol] = '';
                    let safe = true;
                    if (piece === (color === 'white' ? '♔' : '♚')) {
                        // If moving king, check new square
                        safe = !isSquareAttacked(toRow, toCol, color === 'white' ? 'black' : 'white');
                    } else {
                        safe = !isKingInCheck(color);
                    }
                    gameBoard[fromRow][fromCol] = backupFrom;
                    gameBoard[toRow][toCol] = backupTo;
                    if (safe) return true;
                }
            }
        }
    }
    return false;
}
// Move history for real PGN tracking
let moveHistory = [];
let positionHistory = []; // Track board positions for analysis

// Export game as PGN using moveHistory (mainline moves only)
function exportGameAsPGN() {
    if (moveHistory.length === 0) {
        return '[Event "AI Game"]\n[Site "Local"]\n[Result "*"]\n\n';
    }
    let pgnMoves = '';
    for (let i = 0; i < moveHistory.length; i++) {
        if (i % 2 === 0) {
            pgnMoves += `${Math.floor(i/2) + 1}. `;
        }
        pgnMoves += `${moveHistory[i]} `;
    }
    return `[Event "AI Game"]
[Site "Local"]
[Result "*"]

${pgnMoves.trim()}`;
}

// --- Game Outcome Prediction Functions (NEW AI/ML Feature) ---
async function updateOutcomePrediction(fen) {
    try {
        const response = await fetch('http://127.0.0.1:5000/predict-outcome', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fen: fen })
        });
        
        if (response.ok) {
            const data = await response.json();
            displayOutcomePrediction(data);
        } else {
            console.warn('Outcome prediction failed');
        }
    } catch (error) {
        console.warn('Could not get outcome prediction:', error);
    }
}

function displayOutcomePrediction(predictionData) {
    const outcomeElement = document.getElementById('outcomePrediction');
    const confidenceElement = document.getElementById('predictionConfidence');
    
    if (outcomeElement && confidenceElement) {
        outcomeElement.textContent = `🎯 ${predictionData.prediction_text}`;
        confidenceElement.textContent = `📊 Confidence: ${(predictionData.confidence * 100).toFixed(1)}%`;
        
        // Color coding based on prediction
        outcomeElement.className = 'outcome-prediction';
        if (predictionData.prediction === 2) {
            outcomeElement.style.color = '#4CAF50'; // Green for White advantage
        } else if (predictionData.prediction === 0) {
            outcomeElement.style.color = '#f44336'; // Red for Black advantage
        } else {
            outcomeElement.style.color = '#FF9800'; // Orange for balanced
        }
        
        // Show analysis in console for debugging
        console.log('🎯 Game Analysis:', predictionData.analysis);
        console.log('💡 Recommendations:', predictionData.recommendations);
    }
}

// Call prediction on game start
function initializePrediction() {
    const fen = boardToFEN();
    updateOutcomePrediction(fen);
}

// Call backend to get Elo prediction after game ends
async function getEloPrediction() {
    // --- Automatic calculation of blunders, cpl, result ---
    // Blunders: count illegal move attempts (tracked globally)
    const blunders = window.blunderCount || 0;
    // CPL: proxy as number of non-capture moves (for demo)
    const cpl = window.nonCaptureMoves || 0;
    const moves = moveHistory.length;

    // Result: win if black king captured, loss if white king captured, draw if both kings present at end of 80 moves
    let result = 'draw';
    let whiteKing = false, blackKing = false;
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            if (gameBoard[r][c] === '♔') whiteKing = true;
            if (gameBoard[r][c] === '♚') blackKing = true;
        }
    }
    if (!blackKing && whiteKing) result = 'win';
    else if (!whiteKing && blackKing) result = 'loss';
    else if (moves >= 160) result = 'draw';

    // Save game data to backend for training
    try {
        await fetch('http://127.0.0.1:5000/save-game-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ result, blunders, cpl, moves })
        });
    } catch (err) {
        console.warn('Game data save failed:', err);
    }
    try {
        const res = await fetch('http://127.0.0.1:5000/predict-elo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ blunders, cpl, moves })
        });
        const data = await res.json();
        if (data.predicted_elo) {
            // Enhanced Elo prediction - keep logic but remove alert
            const eloRating = Math.round(data.predicted_elo);
            let eloDescription = '';
            
            if (eloRating >= 1400) {
                eloDescription = 'Strong Player 💪';
            } else if (eloRating >= 1200) {
                eloDescription = 'Intermediate Player 📈';
            } else if (eloRating >= 1000) {
                eloDescription = 'Beginner Player 🌱';
            } else {
                eloDescription = 'Learning Player 📚';
            }
            
            // Log to console instead of showing alert
            console.log(`🏆 Your Performance Analysis:`);
            console.log(`⭐ Estimated Elo: ${eloRating}`);
            console.log(`📊 Skill Level: ${eloDescription}`);
            console.log(`📈 Game Stats:`);
            console.log(`• Total Moves: ${moves}`);
            console.log(`• Blunders: ${blunders}`);
            console.log(`• Average CPL: ${cpl}`);
            console.log(`💡 Keep playing to improve your rating!`);
        } else {
            console.log('📊 Not enough data to estimate Elo yet. Play more games to get your rating!');
        }
    } catch (err) {
        console.warn('Failed to contact backend for Elo prediction. Please ensure the Python server is running.');
    }
}

// Chess piece Unicode symbols
const pieces = {
    white: {
        king: '♔',
        queen: '♕',
        rook: '♖',
        bishop: '♗',
        knight: '♘',
        pawn: '♙'
    },
    black: {
        king: '♚',
        queen: '♛',
        rook: '♜',
        bishop: '♝',
        knight: '♞',
        pawn: '♟'
    }
};

// Initial chess board setup
const initialBoard = [
    ['♜', '♞', '♝', '♛', '♚', '♝', '♞', '♜'],
    ['♟', '♟', '♟', '♟', '♟', '♟', '♟', '♟'],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['♙', '♙', '♙', '♙', '♙', '♙', '♙', '♙'],
    ['♖', '♘', '♗', '♕', '♔', '♗', '♘', '♖']
];

let currentPlayer = 'white';
let selectedSquare = null;
let gameBoard = [];
let capturedWhite = [];
let capturedBlack = [];

// Adaptive AI variables
let playerAggressiveScore = 0;
let playerDefensiveScore = 0;
let aiMode = 'default'; // 'aggressive', 'defensive', 'default'

// NEW: Track if this is a new game
let isNewGame = true;
let moveInProgress = false; // Prevent duplicate move processing

// Initialize the game
function initializeGame() {
    gameBoard = initialBoard.map(row => [...row]);
    currentPlayer = 'white';
    selectedSquare = null;
    capturedWhite = [];
    capturedBlack = [];
    moveHistory = []; // Clear move history
    positionHistory = []; // Clear position history
    playerAggressiveScore = 0;
    playerDefensiveScore = 0;
    isNewGame = true; // ADDED: Mark as new game
    window.blunderCount = 0;
    window.nonCaptureMoves = 0;
    
    console.log('🆕 [INIT] Game initialized - moveHistory cleared:', moveHistory);
    
    updateGameInfo();
    createBoard();
    updateCapturedPieces();
    
    // Initialize outcome prediction for new game
    setTimeout(() => {
        initializePrediction();
    }, 500); // Small delay to ensure board is ready
}

// NEW: Add New Game button function
function startNewGame() {
    closeGameEndModal(); // Close modal first
    initializeGame();
    console.log("[INFO] New game started - isNewGame flag set to true");
}

// Game End Modal Functions
function showGameEndModal(winner, endType, gameStats) {
    const modal = document.getElementById('gameEndModal');
    const gameResult = document.getElementById('gameResult');
    const winnerAnnouncement = document.getElementById('winnerAnnouncement');
    const statsElement = document.getElementById('gameStats');
    
    // Determine result emoji and message
    let resultEmoji = '';
    let resultMessage = '';
    let isPlayerWin = false;
    
    if (endType === 'checkmate') {
        if (winner.includes('User')) {
            resultEmoji = '🎉👑🏆';
            resultMessage = 'VICTORY!';
            isPlayerWin = true;
        } else {
            resultEmoji = '😔💔';
            resultMessage = 'DEFEAT';
        }
    } else if (endType === 'stalemate') {
        resultEmoji = '🤝⚖️';
        resultMessage = 'STALEMATE';
    } else if (endType === 'king_captured') {
        if (winner.includes('User')) {
            resultEmoji = '🎯👑💥';
            resultMessage = 'KING CAPTURED!';
            isPlayerWin = true;
        } else {
            resultEmoji = '☠️👑';
            resultMessage = 'KING LOST!';
        }
    } else if (endType === 'insufficient_material') {
        resultEmoji = '🏆⚔️👑';
        resultMessage = 'VICTORY! INSUFFICIENT MATERIAL';
        isPlayerWin = true;
    } else if (endType === 'move_limit') {
        resultEmoji = '⏰🤝';
        resultMessage = 'TIME DRAW';
    }
    
    // Set content
    gameResult.textContent = resultEmoji;
    winnerAnnouncement.textContent = resultMessage;
    
    // Add victory/defeat animation classes
    if (isPlayerWin) {
        winnerAnnouncement.className = 'winner-announcement victory-player';
    } else if (winner.includes('AI')) {
        winnerAnnouncement.className = 'winner-announcement victory-ai';
    } else {
        winnerAnnouncement.className = 'winner-announcement';
    }
    
    // Show game statistics
    const moves = moveHistory.length;
    const capturedByPlayer = capturedBlack.length;
    const capturedByAI = capturedWhite.length;
    const blunders = window.blunderCount || 0;
    
    statsElement.innerHTML = `
        <div><strong>🎮 Game Statistics</strong></div>
        <div>📋 Total Moves: ${moves}</div>
        <div>🏹 You Captured: ${capturedByPlayer} pieces</div>
        <div>🤖 AI Captured: ${capturedByAI} pieces</div>
        <div>⚠️ Blunders: ${blunders}</div>
        <div>🏆 Winner: ${winner}</div>
    `;
    
    // Show alert instead of modal
    alert(`${resultEmoji}

${resultMessage}

🎮 Game Statistics:
📋 Total Moves: ${moves}
🏹 You Captured: ${capturedByPlayer} pieces
🤖 AI Captured: ${capturedByAI} pieces
⚠️ Blunders: ${blunders}

🏆 Winner: ${winner}`);
    
// Show immediate feedback if player lost
    if (winner.includes('AI')) {
        console.log('🏆 [DEFEAT] Player lost to AI - showing feedback...');
        console.log('📋 [DEFEAT] Current moveHistory:', moveHistory);
        console.log('📊 [DEFEAT] moveHistory length:', moveHistory.length);
        
        if (moveHistory.length === 0) {
            console.warn('⚠️ [DEFEAT] No moves in history - cannot provide feedback!');
            setTimeout(() => {
                alert('🎯 GAME TOO SHORT\n\nNo moves to analyze!\n\nPlay a longer game to get move feedback.');
            }, 1000);
        } else {
            showImmediateFeedback();
        }
    }
    
    // Play victory/defeat sound (optional - you can add sound files)
    if (isPlayerWin) {
        console.log('🎉 Player Victory! Well played!');
    } else if (winner.includes('AI')) {
        console.log('🤖 AI Victory! Better luck next time!');
    }
}

function closeGameEndModal() {
    const modal = document.getElementById('gameEndModal');
    modal.style.display = 'none';
}

// Show immediate feedback alert after defeat
function showImmediateFeedback() {
    console.log('📚 [IMMEDIATE] Getting quick feedback...');
    
    const currentFEN = boardToFEN();
    console.log('📍 [IMMEDIATE] Current FEN:', currentFEN);
    console.log('📋 [IMMEDIATE] moveHistory to send:', moveHistory);
    console.log('📊 [IMMEDIATE] Total moves to analyze:', moveHistory.length);
    
    if (moveHistory.length === 0) {
        console.error('❌ [IMMEDIATE] No moves in history - cannot analyze!');
        alert('🎯 NO MOVES TO ANALYZE\n\nGame ended too quickly!\nPlay a longer game to get feedback.');
        return;
    }
    
    const requestData = {
        move_history: moveHistory,
        final_fen: currentFEN
    };
    
    console.log('📦 [IMMEDIATE] Sending request data:', requestData);
    
    // Get quick feedback from backend
    fetch('http://127.0.0.1:5000/analyze-game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        console.log('🔍 [IMMEDIATE] Response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('📊 [IMMEDIATE] Got analysis data:', data);
        
        if (data.success && data.feedback && data.feedback.length > 0) {
            // Create immediate feedback alert
            let feedbackText = '🎯 BETTER MOVES YOU COULD HAVE MADE:\n\n';
            
            // Show top 3 feedback suggestions
            const topFeedback = data.feedback.slice(0, 3);
            
            topFeedback.forEach((analysis, index) => {
                if (analysis.alternatives && analysis.alternatives[0]) {
                    const bestAlt = analysis.alternatives[0];
                    const improvement = bestAlt.evaluation - analysis.your_eval;
                    
                    feedbackText += `${index + 1}. Move ${analysis.move_number}:\n`;
                    feedbackText += `   ❌ Your move: ${analysis.your_move}\n`;
                    feedbackText += `   ✅ Better: ${bestAlt.san} (+${improvement.toFixed(1)})\n\n`;
                }
            });
            
            feedbackText += '💪 Learn from these moves to improve!\n';
            feedbackText += '📊 Check console for detailed analysis!\n\n';
            feedbackText += '🎮 Click OK to continue';
            
            // Show the feedback alert IMMEDIATELY
            console.log('⚡ [IMMEDIATE] Showing feedback alert NOW!');
            alert(feedbackText);
            console.log('✅ [IMMEDIATE] Feedback alert dismissed by user');
            
        } else {
            // No specific feedback, show general tips IMMEDIATELY
            console.log('💡 [IMMEDIATE] No specific feedback, showing general tips');
            alert('🎯 GAME ANALYSIS\n\nNo major tactical errors found!\n\n💡 General Tips:\n• Develop pieces early\n• Control the center\n• Keep king safe\n• Look for tactics\n• Think before moving\n\n💪 Keep practicing!');
        }
        
        // Also trigger detailed console analysis
        showGameFeedback();
        
    })
    .catch(error => {
        console.error('❌ [IMMEDIATE] Feedback failed:', error);
        
        // Show offline feedback IMMEDIATELY
        console.log('🔌 [IMMEDIATE] Offline mode - showing quick tips');
        alert('🎯 QUICK TIPS TO IMPROVE:\n\n• Look for checks, captures, threats\n• Develop pieces before attacking\n• Control center squares (d4, e4, d5, e5)\n• Castle early for king safety\n• Don\'t move same piece twice in opening\n\n💪 Practice makes perfect!');
        
        // Still try detailed feedback
        showGameFeedback();
    });
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('gameEndModal');
    if (event.target == modal) {
        closeGameEndModal();
    }
}

// Create the chess board
function createBoard() {
    const board = document.getElementById('chessBoard');
    board.innerHTML = '';

    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const square = document.createElement('div');
            square.classList.add('square');
            square.classList.add((row + col) % 2 === 0 ? 'light' : 'dark');
            square.dataset.row = row;
            square.dataset.col = col;
            
            const piece = gameBoard[row][col];
            if (piece) {
                const pieceElement = document.createElement('div');
                pieceElement.classList.add('piece');
                pieceElement.textContent = piece;
                pieceElement.draggable = true;
                square.appendChild(pieceElement);
            }

            square.addEventListener('click', handleSquareClick);
            square.addEventListener('dragover', handleDragOver);
            square.addEventListener('drop', handleDrop);
            
            if (square.querySelector('.piece')) {
                square.querySelector('.piece').addEventListener('dragstart', handleDragStart);
                square.querySelector('.piece').addEventListener('dragend', handleDragEnd);
            }

            board.appendChild(square);
        }
    }
}

// Handle square clicks
function handleSquareClick(e) {
    const square = e.currentTarget;
    const row = parseInt(square.dataset.row);
    const col = parseInt(square.dataset.col);
    
    if (selectedSquare) {
        if (selectedSquare.row === row && selectedSquare.col === col) {
            // Deselect if clicking the same square
            clearSelection();
        } else if (isValidMove(selectedSquare.row, selectedSquare.col, row, col)) {
            // Make the move
            console.log(`[CLICK DEBUG] 💆 Making move: (${selectedSquare.row},${selectedSquare.col}) -> (${row},${col})`);
            makeMove(selectedSquare.row, selectedSquare.col, row, col);
            clearSelection();
            switchPlayer();
        } else {
            // Select new piece if it belongs to current player
            const piece = gameBoard[row][col];
            if (piece && isPieceOwnedByCurrentPlayer(piece)) {
                selectSquare(row, col);
            } else {
                clearSelection();
            }
        }
    } else {
        // Select a piece if it belongs to current player
        const piece = gameBoard[row][col];
        if (piece && isPieceOwnedByCurrentPlayer(piece)) {
            selectSquare(row, col);
        }
    }
}

// Select a square
function selectSquare(row, col) {
    clearSelection();
    selectedSquare = { row, col };
    const square = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
    square.classList.add('selected');
    highlightValidMoves(row, col);
}

// Clear selection
function clearSelection() {
    selectedSquare = null;
    document.querySelectorAll('.square').forEach(square => {
        square.classList.remove('selected', 'valid-move');
    });
}

// Highlight valid moves
function highlightValidMoves(row, col) {
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            if (isValidMove(row, col, r, c)) {
                const square = document.querySelector(`[data-row="${r}"][data-col="${c}"]`);
                square.classList.add('valid-move');
            }
        }
    }
}

// Check if piece belongs to current player
function isPieceOwnedByCurrentPlayer(piece) {
    const whitePieces = Object.values(pieces.white);
    const blackPieces = Object.values(pieces.black);
    
    if (currentPlayer === 'white') {
        return whitePieces.includes(piece);
    } else {
        return blackPieces.includes(piece);
    }
}

// Basic move validation (simplified)
function isValidMove(fromRow, fromCol, toRow, toCol) {
    // Can't move to same position
    if (fromRow === toRow && fromCol === toCol) return false;
    // Can't move outside board
    if (toRow < 0 || toRow > 7 || toCol < 0 || toCol > 7) return false;
    const piece = gameBoard[fromRow][fromCol];
    const targetPiece = gameBoard[toRow][toCol];
    // Can't capture own piece
    if (targetPiece && isPieceOwnedByCurrentPlayer(targetPiece)) return false;
    // Basic movement rules (simplified - not covering all chess rules)
    const rowDiff = Math.abs(toRow - fromRow);
    const colDiff = Math.abs(toCol - fromCol);
    // Pawn movement
    if (piece === '♙' || piece === '♟') {
        const direction = piece === '♙' ? -1 : 1;
        const startRow = piece === '♙' ? 6 : 1;
        // Forward move
        if (fromCol === toCol && !targetPiece) {
            if (toRow === fromRow + direction) return true;
            if (fromRow === startRow && toRow === fromRow + 2 * direction) return true;
        }
        // Diagonal capture
        if (Math.abs(fromCol - toCol) === 1 && toRow === fromRow + direction && targetPiece) {
            return true;
        }
        return false;
    }
    // Rook movement
    if (piece === '♖' || piece === '♜') {
        if (fromRow === toRow || fromCol === toCol) {
            return isPathClear(fromRow, fromCol, toRow, toCol);
        }
        return false;
    }
    // Bishop movement
    if (piece === '♗' || piece === '♝') {
        if (rowDiff === colDiff) {
            return isPathClear(fromRow, fromCol, toRow, toCol);
        }
        return false;
    }
    // Queen movement
    if (piece === '♕' || piece === '♛') {
        if (fromRow === toRow || fromCol === toCol || rowDiff === colDiff) {
            return isPathClear(fromRow, fromCol, toRow, toCol);
        }
        return false;
    }
    // King movement
    if (piece === '♔' || piece === '♚') {
        return rowDiff <= 1 && colDiff <= 1;
    }
    // Knight movement
    if (piece === '♘' || piece === '♞') {
        return (rowDiff === 2 && colDiff === 1) || (rowDiff === 1 && colDiff === 2);
    }
    return false;
}

// Check if path is clear for pieces that move in lines
function isPathClear(fromRow, fromCol, toRow, toCol) {
    const rowStep = toRow > fromRow ? 1 : toRow < fromRow ? -1 : 0;
    const colStep = toCol > fromCol ? 1 : toCol < fromCol ? -1 : 0;
    let currentRow = fromRow + rowStep;
    let currentCol = fromCol + colStep;
    while (currentRow !== toRow || currentCol !== toCol) {
        if (gameBoard[currentRow][currentCol] !== '') {
            return false;
        }
        currentRow += rowStep;
        currentCol += colStep;
    }
    return true;
}

// Make a move (no animation)
function makeMove(fromRow, fromCol, toRow, toCol) {
    // CRITICAL: Prevent duplicate processing
    if (moveInProgress) {
        console.warn(`[MOVE WARNING] ⚠️ Move already in progress - ignoring duplicate call`);
        return;
    }
    
    moveInProgress = true; // Lock to prevent duplicates
    
    try {
        // CRITICAL VALIDATION: Prevent any invalid moves from being recorded
        console.log(`\n=== MAKEMOVE DEBUG ===`);
        console.log(`Input coordinates: from(${fromRow},${fromCol}) to(${toRow},${toCol})`);
        
        // Validate input parameters are numbers
        if (typeof fromRow !== 'number' || typeof fromCol !== 'number' || 
            typeof toRow !== 'number' || typeof toCol !== 'number') {
            console.error(`[MOVE ERROR] Non-numeric coordinates:`, {fromRow, fromCol, toRow, toCol});
            return;
        }
        
        // Validate coordinates are within bounds
        if (fromCol < 0 || fromCol > 7 || toCol < 0 || toCol > 7 ||
            fromRow < 0 || fromRow > 7 || toRow < 0 || toRow > 7) {
            console.error(`[MOVE ERROR] Coordinates out of bounds: from(${fromRow},${fromCol}) to(${toRow},${toCol})`);
            return;
        }
        
        // Check if move makes sense (not same square)
        if (fromRow === toRow && fromCol === toCol) {
            console.error(`[MOVE ERROR] Same square move attempt`);
            return;
        }
        
        // Validate there's actually a piece to move
        const piece = gameBoard[fromRow][fromCol];
        if (!piece || piece === '') {
            console.error(`[MOVE ERROR] No piece at source square (${fromRow},${fromCol})`);
            return;
        }
        
        // Validate the move is actually legal according to chess rules
        if (!isValidMove(fromRow, fromCol, toRow, toCol)) {
            console.error(`[MOVE ERROR] Invalid move according to chess rules: ${piece} from (${fromRow},${fromCol}) to (${toRow},${toCol})`);
            return;
        }
        
        console.log(`Valid move confirmed: ${piece} from (${fromRow},${fromCol}) to (${toRow},${toCol})`);
        
        const capturedPiece = gameBoard[toRow][toCol];
        // Handle captured piece
        if (capturedPiece) {
            if (isPieceOwnedByCurrentPlayer(capturedPiece)) return; // Can't capture own piece
            if (Object.values(pieces.white).includes(capturedPiece)) {
                capturedWhite.push(capturedPiece);
            } else {
                capturedBlack.push(capturedPiece);
            }
        }
        // Move the piece
        gameBoard[toRow][toCol] = piece;
        gameBoard[fromRow][fromCol] = '';

        // Record move in UCI format: like 'e2e4'
        const moveStr = String.fromCharCode(97 + fromCol) + (8 - fromRow) +
                    String.fromCharCode(97 + toCol) + (8 - toRow);
    
        // Final UCI validation
        const fromSquare = String.fromCharCode(97 + fromCol) + (8 - fromRow);
        const toSquare = String.fromCharCode(97 + toCol) + (8 - toRow);
        
        // Validate UCI format
        if (moveStr.length !== 4 || fromSquare === toSquare) {
            console.error(`[MOVE ERROR] Invalid UCI format: ${moveStr}`);
            return;
        }
        
        // Check if this exact move was just played (prevent duplicates)
        if (moveHistory.length > 0 && moveHistory[moveHistory.length - 1] === moveStr) {
            console.error(`[MOVE ERROR] Duplicate move prevented: ${moveStr}`);
            return;
        }
        
        // SUCCESS: Record the validated move
        moveHistory.push(moveStr);
        
        console.log(`✅ MOVE RECORDED: ${moveStr} (${fromSquare} to ${toSquare})`);
        console.log(`📋 Updated moveHistory (${moveHistory.length}):`, moveHistory);
        console.log(`=== END MAKEMOVE DEBUG ===\n`);

        // --- Check for king capture only (no insufficient material check) ---
        let whiteKing = false, blackKing = false;
        
        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const boardPiece = gameBoard[r][c];
                if (boardPiece) {
                    if (boardPiece === '♔') whiteKing = true;
                    if (boardPiece === '♚') blackKing = true;
                }
            }
        }
        
        // Only end game if king is actually captured, not just isolated
        if (!whiteKing || !blackKing) {
            console.log('[GAME END] King captured');
            setTimeout(() => {
                const winner = whiteKing ? 'User (White)' : 'AI (Black)';
                showGameEndModal(winner, 'king_captured', {
                    moves: moveHistory.length,
                    capturedByPlayer: capturedBlack.length,
                    capturedByAI: capturedWhite.length,
                    blunders: window.blunderCount || 0
                });
                getEloPrediction();
            }, 500);
            return;
        }

        // --- Adaptive AI: Track player style ---
        if (currentPlayer === 'white') {
            if (capturedPiece) {
                playerAggressiveScore++;
            } else {
                // If move is not a capture and piece moves to safe square, consider defensive
                if (!isSquareAttacked(toRow, toCol, 'black')) playerDefensiveScore++;
            }
            updateAIMode();
        }
        createBoard();
        updateCapturedPieces();
    } finally {
        // Always release the lock, even if an error occurs
        setTimeout(() => {
            moveInProgress = false;
        }, 100);
    }
}

// Check if a square is attacked by opponent (for defensive score)
function isSquareAttacked(row, col, attackerColor) {
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const piece = gameBoard[r][c];
            if (piece && ((attackerColor === 'white' && Object.values(pieces.white).includes(piece)) ||
                (attackerColor === 'black' && Object.values(pieces.black).includes(piece)))) {
                if (isValidMove(r, c, row, col)) return true;
            }
        }
    }
    return false;
}

// Update AI mode based on player style
function updateAIMode() {
    if (playerAggressiveScore - playerDefensiveScore > 2) {
        aiMode = 'defensive';
    } else if (playerDefensiveScore - playerAggressiveScore > 2) {
        aiMode = 'aggressive';
    } else {
        aiMode = 'default';
    }
}

// Switch current player
function switchPlayer() {
    currentPlayer = currentPlayer === 'white' ? 'black' : 'white';
    updateGameInfo();
    // If it's AI's turn, call backend for move
    if (currentPlayer === 'black') {
        setTimeout(aiMove, 500);
    }
}

// --- UPDATED AI Move via Python Backend ---
async function aiMove() {
    // Convert board to FEN
    const fen = boardToFEN();
    
    // Get last player move (if any)
    const lastMove = moveHistory.length > 0 ? moveHistory[moveHistory.length - 1] : null;
    
    console.log(`[DEBUG] AI Move Request - isNewGame: ${isNewGame}, lastMove: ${lastMove}, fen: ${fen}`);
    
    try {
        const res = await fetch('http://127.0.0.1:5000/ai-move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                fen: fen,
                last_move: lastMove,
                is_new_game: isNewGame  // ADDED: Send new game flag
            })
        });
        const data = await res.json();
        
        // Log the AI response
        console.log("[DEBUG] AI Response:", data);
        
        // Check if game is over from backend
        if (data.game_over) {
            console.log("[GAME OVER] Backend detected game end:", data.reason);
            
            let winner = '';
            let endType = data.reason;
            
            if (data.winner === 'white') {
                winner = 'User (White)';
            } else if (data.winner === 'black') {
                winner = 'AI (Black)';
            } else {
                winner = 'Draw';
            }
            
            // Show game end modal
            setTimeout(() => {
                showGameEndModal(winner, endType, {
                    moves: moveHistory.length,
                    capturedByPlayer: capturedBlack.length,
                    capturedByAI: capturedWhite.length,
                    blunders: window.blunderCount || 0
                });
                getEloPrediction();
            }, 1000);
            
            return;
        }
        
        if (data.move && data.fen) {
            // CRITICAL: Validate AI move before recording
            console.log(`\n=== AI MOVE VALIDATION ===`);
            console.log(`AI returned move: ${data.move}`);
            
            // Validate UCI format
            if (typeof data.move !== 'string' || data.move.length !== 4) {
                console.error(`[AI ERROR] Invalid move format: ${data.move}`);
                return;
            }
            
            // Validate move is not duplicate
            if (moveHistory.length > 0 && moveHistory[moveHistory.length - 1] === data.move) {
                console.error(`[AI ERROR] Duplicate move prevented: ${data.move}`);
                return;
            }
            
            // Parse UCI coordinates
            const fromCol = data.move.charCodeAt(0) - 97;  // a-h -> 0-7
            const fromRow = 8 - parseInt(data.move[1]);     // 1-8 -> 7-0
            const toCol = data.move.charCodeAt(2) - 97;     // a-h -> 0-7  
            const toRow = 8 - parseInt(data.move[3]);       // 1-8 -> 7-0
            
            console.log(`Parsed AI move: from(${fromRow},${fromCol}) to(${toRow},${toCol})`);
            
            // Validate coordinates are in bounds
            if (fromCol < 0 || fromCol > 7 || toCol < 0 || toCol > 7 ||
                fromRow < 0 || fromRow > 7 || toRow < 0 || toRow > 7 ||
                isNaN(fromRow) || isNaN(fromCol) || isNaN(toRow) || isNaN(toCol)) {
                console.error(`[AI ERROR] Invalid coordinates in move ${data.move}`);
                return;
            }
            
            // SUCCESS: Record the validated AI move
            moveHistory.push(data.move);
            console.log(`✅ AI MOVE RECORDED: ${data.move}`);
            console.log(`📋 Updated moveHistory (${moveHistory.length}):`, moveHistory);
            console.log(`=== END AI MOVE VALIDATION ===\n`);
            console.log(`[AI MOVE] Updated moveHistory length: ${moveHistory.length}`);
            
            // Update board from FEN
            fenToBoard(data.fen);
            createBoard();
            updateCapturedPieces();
            currentPlayer = 'white';
            updateGameInfo();

            // Check for game end after AI move (for human)
            let whiteKing = false, blackKing = false;
            
            for (let r = 0; r < 8; r++) {
                for (let c = 0; c < 8; c++) {
                    const boardPiece = gameBoard[r][c];
                    if (boardPiece) {
                        if (boardPiece === '♔') whiteKing = true;
                        if (boardPiece === '♚') blackKing = true;
                    }
                }
            }
            
            let gameEnded = false;
            let endType = '';
            let winner = '';
            
            // Check for king capture or move limit (NO insufficient material check)
            if (!whiteKing || !blackKing) {
                gameEnded = true;
                endType = 'king_captured';
                winner = whiteKing ? 'User (White)' : 'AI (Black)';
            } else if (moveHistory.length >= 160) {
                gameEnded = true;
                endType = 'move_limit';
                winner = 'Draw';
            } else {
                // Checkmate or stalemate for human
                const noLegal = !hasAnyLegalMove('white');
                if (noLegal) {
                    if (isKingInCheck('white')) {
                        endType = 'checkmate';
                        winner = 'AI (Black)';
                    } else {
                        endType = 'stalemate';
                        winner = 'Draw';
                    }
                    gameEnded = true;
                }
            }
            // ADDED: Clear new game flag after first AI move
            if (isNewGame) {
                isNewGame = false;
                console.log("[INFO] New game flag cleared after first AI move");
            }
            
            // Update outcome prediction after AI move
            const newFen = boardToFEN();
            updateOutcomePrediction(newFen);
            
            // Display AI info (optional)
            if (data.engine || data.profile || data.outcome_prediction) {
                console.log(`[AI INFO] Engine: ${data.engine}, Profile: ${data.profile}, Prediction: ${data.outcome_prediction}`);
            }
            if (gameEnded) {
                let msg = '';
                if (endType === 'checkmate') {
                    msg = `Checkmate! Winner: ${winner}`;
                } else if (endType === 'stalemate') {
                    msg = 'Stalemate! Game is a draw.';
                } else if (endType === 'king_captured') {
                    msg = `King captured! Winner: ${winner}`;
                } else if (endType === 'insufficient_material') {
                    msg = `Insufficient material! Winner: ${winner}`;
                } else if (endType === 'move_limit') {
                    msg = '80-move limit reached! Game is a draw.';
                }
                
                // Update game status
                updateGameStatus(`Game Over: ${msg}`);
                
                // Show enhanced game end modal instead of alert
                setTimeout(() => {
                    showGameEndModal(winner, endType, {
                        moves: moveHistory.length,
                        capturedByPlayer: capturedBlack.length,
                        capturedByAI: capturedWhite.length,
                        blunders: window.blunderCount || 0
                    });
                    getEloPrediction();
                    
                    // Show feedback if user lost
                    if (winner.includes('AI')) {
                        setTimeout(() => {
                            showGameFeedback();
                        }, 2000);
                    }
                }, 1000); // Small delay for better UX
            }
        } else {
            alert('AI error: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('AI Backend Error:', err);
        alert('Failed to contact AI backend. Is Python server running?');
    }
}

// Convert board to FEN (simple, no castling/en passant)
function boardToFEN() {
    let fen = '';
    for (let r = 0; r < 8; r++) {
        let empty = 0;
        for (let c = 0; c < 8; c++) {
            const p = gameBoard[r][c];
            if (!p) {
                empty++;
            } else {
                if (empty > 0) { fen += empty; empty = 0; }
                fen += pieceToFEN(p);
            }
        }
        if (empty > 0) fen += empty;
        if (r < 7) fen += '/';
    }
    fen += ' ' + (currentPlayer === 'white' ? 'w' : 'b') + ' - - 0 1';
    return fen;
}

function pieceToFEN(p) {
    switch (p) {
        case '♔': return 'K'; case '♕': return 'Q'; case '♖': return 'R'; case '♗': return 'B'; case '♘': return 'N'; case '♙': return 'P';
        case '♚': return 'k'; case '♛': return 'q'; case '♜': return 'r'; case '♝': return 'b'; case '♞': return 'n'; case '♟': return 'p';
        default: return '';
    }
}

// Update board from FEN
function fenToBoard(fen) {
    const rows = fen.split(' ')[0].split('/');
    for (let r = 0; r < 8; r++) {
        let row = [];
        let i = 0;
        for (const ch of rows[r]) {
            if (!isNaN(ch)) {
                for (let k = 0; k < parseInt(ch); k++) row.push('');
            } else {
                row.push(fenToPiece(ch));
            }
        }
        gameBoard[r] = row;
    }
}

function fenToPiece(ch) {
    switch (ch) {
        case 'K': return '♔'; case 'Q': return '♕'; case 'R': return '♖'; case 'B': return '♗'; case 'N': return '♘'; case 'P': return '♙';
        case 'k': return '♚'; case 'q': return '♛'; case 'r': return '♜'; case 'b': return '♝'; case 'n': return '♞'; case 'p': return '♟';
        default: return '';
    }
}

// Update game information
function updateGameInfo() {
    document.getElementById('currentPlayer').textContent = `Current Player: ${currentPlayer.charAt(0).toUpperCase() + currentPlayer.slice(1)}`;
    document.getElementById('gameStatus').textContent = 'Game in progress';
}

// Update game status for game end
function updateGameStatus(status) {
    document.getElementById('gameStatus').textContent = status;
    document.getElementById('gameStatus').style.color = '#e74c3c';
    document.getElementById('gameStatus').style.fontWeight = 'bold';
}

// Update captured pieces display
function updateCapturedPieces() {
    document.getElementById('capturedWhite').textContent = capturedWhite.join(' ');
    document.getElementById('capturedBlack').textContent = capturedBlack.join(' ');
}

// Drag and drop handlers
function handleDragStart(e) {
    const piece = e.target.textContent;
    if (!isPieceOwnedByCurrentPlayer(piece)) {
        e.preventDefault();
        return;
    }
    e.target.classList.add('dragging');
    const square = e.target.parentElement;
    selectedSquare = {
        row: parseInt(square.dataset.row),
        col: parseInt(square.dataset.col)
    };
    highlightValidMoves(selectedSquare.row, selectedSquare.col);
}

function handleDragEnd(e) {
    e.target.classList.remove('dragging');
    clearSelection();
}

function handleDragOver(e) {
    e.preventDefault();
}

function handleDrop(e) {
    e.preventDefault();
    if (!selectedSquare) return;
    const square = e.currentTarget;
    const toRow = parseInt(square.dataset.row);
    const toCol = parseInt(square.dataset.col);
    if (isValidMove(selectedSquare.row, selectedSquare.col, toRow, toCol)) {
        console.log(`[DRAG DEBUG] 👜 Making move: (${selectedSquare.row},${selectedSquare.col}) -> (${toRow},${toCol})`);
        makeMove(selectedSquare.row, selectedSquare.col, toRow, toCol);
        // --- Check for game end: checkmate, stalemate, king missing, or 80+ moves ---
        let whiteKing = false, blackKing = false;
        
        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const boardPiece = gameBoard[r][c];
                if (boardPiece) {
                    if (boardPiece === '♔') whiteKing = true;
                    if (boardPiece === '♚') blackKing = true;
                }
            }
        }
        
        let gameEnded = false;
        let endType = '';
        let winner = '';
        
        // Check for king capture or move limit (NO insufficient material check)
        if (!whiteKing || !blackKing) {
            gameEnded = true;
            endType = 'king_captured';
            winner = whiteKing ? 'User (White)' : 'AI (Black)';
        } else if (moveHistory.length >= 160) {
            gameEnded = true;
            endType = 'move_limit';
            winner = 'Draw';
        } else {
            // Checkmate or stalemate
            const noLegal = !hasAnyLegalMove(currentPlayer);
            if (noLegal) {
                if (isKingInCheck(currentPlayer)) {
                    endType = 'checkmate';
                    winner = currentPlayer === 'white' ? 'AI (Black)' : 'User (White)';
                } else {
                    endType = 'stalemate';
                    winner = 'Draw';
                }
                gameEnded = true;
            }
        }
        if (gameEnded) {
            let msg = '';
            if (endType === 'checkmate') {
                msg = `Checkmate! Winner: ${winner}`;
            } else if (endType === 'stalemate') {
                msg = 'Stalemate! Game is a draw.';
            } else if (endType === 'king_captured') {
                msg = `King captured! Winner: ${winner}`;
            } else if (endType === 'insufficient_material') {
                msg = `Insufficient material! Winner: ${winner}`;
            } else if (endType === 'move_limit') {
                msg = '160-move limit reached! Game is a draw.';
            }
            
            // Show enhanced game end modal instead of alert
            setTimeout(() => {
                showGameEndModal(winner, endType, {
                    moves: moveHistory.length,
                    capturedByPlayer: capturedBlack.length,
                    capturedByAI: capturedWhite.length,
                    blunders: window.blunderCount || 0
                });
                getEloPrediction();
                
                // Show feedback if user lost
                if (winner.includes('AI')) {
                    setTimeout(() => {
                        showGameFeedback();
                    }, 2000);
                }
            }, 1000); // Small delay for better UX
        } else {
            switchPlayer();
        }
    } else {
        // Track blunders as illegal move attempts
        if (!window.blunderCount) window.blunderCount = 0;
        window.blunderCount++;
    }
    clearSelection();
}

// Initialize the game when page loads
document.addEventListener('DOMContentLoaded', initializeGame);

// Feedback system after game loss
async function showGameFeedback() {
    console.log('🎯 [FEEDBACK] Starting game analysis...');
    console.log('📋 [FEEDBACK] Total moves:', moveHistory.length);
    
    try {
        const fen = boardToFEN();
        console.log('📍 [FEEDBACK] Sending analysis request to backend...');
        
        const response = await fetch('http://127.0.0.1:5000/analyze-game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                move_history: moveHistory,
                final_fen: fen
            })
        });
        
        console.log('✅ [FEEDBACK] Backend response received');
        
        if (response.ok) {
            const data = await response.json();
            console.log('📊 [FEEDBACK] Analysis data:', data);
            displayFeedback(data.feedback);
        } else {
            console.warn('⚠️ [FEEDBACK] Game analysis failed - response not OK');
        }
    } catch (error) {
        console.error('❌ [FEEDBACK] Could not get game feedback:', error);
    }
}

function displayFeedback(feedback) {
    if (!feedback || feedback.length === 0) {
        console.log('No feedback available');
        return;
    }
    
    let feedbackHTML = `
        <div style="background: white; padding: 20px; border-radius: 10px; max-width: 600px; margin: 20px auto;">
            <h2 style="color: #667eea; margin-bottom: 15px;">📊 Game Analysis - Better Moves You Could Have Played</h2>
    `;
    
    feedback.forEach(move => {
        feedbackHTML += `
            <div style="margin: 15px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; border-left: 4px solid #764ba2;">
                <div style="font-weight: bold; color: #333; margin-bottom: 10px;">Move ${move.move_number}</div>
                <div style="color: #666; margin-bottom: 8px;">You played: <strong>${move.your_move}</strong> (Eval: ${move.your_eval})</div>
                <div style="color: #4CAF50; font-weight: bold; margin-bottom: 5px;">Better alternatives:</div>
                <ul style="margin-left: 20px; color: #555;">
        `;
        
        move.alternatives.forEach((alt, index) => {
            const improvement = alt.evaluation - move.your_eval;
            feedbackHTML += `
                <li style="margin: 5px 0;">
                    <strong>${alt.san}</strong> (${alt.move}) - Eval: ${alt.evaluation} 
                    <span style="color: ${improvement > 0 ? '#4CAF50' : '#f44336'};">(
                        ${improvement > 0 ? '+' : ''}${improvement.toFixed(1)}
                    )</span>
                </li>
            `;
        });
        
        feedbackHTML += `
                </ul>
            </div>
        `;
    });
    
    feedbackHTML += `
            <button onclick="this.parentElement.remove()" 
                style="padding: 10px 20px; background: #667eea; color: white; border: none; 
                border-radius: 5px; cursor: pointer; margin-top: 15px;">
                Close Feedback
            </button>
        </div>
    `;
    
    // Display feedback modal
    const feedbackDiv = document.createElement('div');
    feedbackDiv.innerHTML = feedbackHTML;
    feedbackDiv.style.cssText = `
        position: fixed; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 100%; 
        background: rgba(0,0,0,0.8); 
        display: flex; 
        justify-content: center; 
        align-items: center; 
        z-index: 10000;
        overflow-y: auto;
        padding: 20px;
    `;
    document.body.appendChild(feedbackDiv);
    
    console.log('📊 Feedback displayed:', feedback);
}