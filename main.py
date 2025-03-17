import os
import json
import random
from typing import Dict, List, Optional
from contextlib import contextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from uuid import uuid4
from datetime import datetime

# Import database connection function
from db_connect import connect_to_db

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize FastAPI app
app = FastAPI(title="Hangman Game API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# ======================================
# 🚀 MODELS
# ======================================
class UserCreate(BaseModel):
    email: str
    password: str
    username: str

class UserLogin(BaseModel):
    email: str
    password: str

class GameState(BaseModel):
    id: str
    word: str
    guessed_letters: List[str] = []
    attempts_left: int
    status: str
    wrong_guesses: List[str] = []
    correct_guesses: List[str] = []
    created_by: str
    created_at: str
    hints: List[str] = []

class PublicGameState(BaseModel):
    id: str
    masked_word: str
    guessed_letters: List[str] = []
    attempts_left: int
    status: str
    wrong_guesses: List[str] = []
    correct_guesses: List[str] = []
    created_by: str
    created_at: str
    hints: List[str] = []

class PublicGameResponse(BaseModel):
    game_id: str
    status: str
    created_by: str
    created_at: str
    attempts_left: int

class LetterGuess(BaseModel):
    letter: str

class CreateGame(BaseModel):
    word: str

class HintRequest(BaseModel):
    hint: str

class RefreshToken(BaseModel):
    refresh_token: str

class RefreshResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int
    user: dict

# ======================================
# 🔧 UTILITY FUNCTIONS
# ======================================
def mask_word(word: str, guessed_letters: List[str]) -> str:
    """Mask the word, replacing unguessed letters with asterisks (*) but keeping spaces visible."""
    return "".join([char if char.lower() in guessed_letters or char == " " else "*" for char in word])

def generate_random_number_between_1000_9999() -> str:
    """Generate a random number between 1000 and 9999 and return as string."""
    return str(random.randint(1000, 9999))

# ======================================
# 🔒 AUTHENTICATION HELPERS
# ======================================
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        response = supabase.auth.get_user(token)
        if not hasattr(response, "user") or not response.user:
            raise Exception("Invalid user data from Supabase")
        
        class UserResponse:
            def __init__(self, id, email, username):
                self.id = id
                self.email = email
                self.username = username
        
        # Extract username from user metadata or profile
        username = response.user.user_metadata.get("username", "Unknown")  # Adjust the key as needed
        
        # Pass id, email, and username to UserResponse
        return UserResponse(response.user.id, response.user.email, username)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

# ======================================
# 🛢️ DATABASE CONNECTION & QUERY EXECUTION
# ======================================
db_connection = None

@contextmanager
def get_db_cursor():
    """Context manager for database cursor"""
    global db_connection
    if db_connection is None or db_connection.closed:
        db_connection = connect_to_db()
        if db_connection is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to connect to database")

    cursor = db_connection.cursor()
    try:
        yield cursor
        db_connection.commit()
    except Exception:
        db_connection.rollback()
        raise e
    finally:
        cursor.close()

def execute_query(query: str, params: Optional[tuple] = None):
    """Execute a database query and return results"""
    with get_db_cursor() as cursor:
        cursor.execute(query, params or ())
        try:
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return [dict(zip(column_names, row)) for row in results]
        except:
            return []

# ======================================
# 🔥 AUTH ROUTES
# ======================================
@app.post("/auth/register")
async def register(user: UserCreate):
    try:
        return supabase.auth.sign_up({
            "email": user.email,
            "password": user.password,
            "options": {"data": {"username": user.username}}
        })
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/auth/login")
async def login(user: UserLogin):
    try:
        return supabase.auth.sign_in_with_password({"email": user.email, "password": user.password})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/auth/logout")
async def logout(user=Depends(get_current_user)):
    try:
        supabase.auth.sign_out()
        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/auth/refresh", response_model=RefreshResponse)
async def refresh_token(refresh_request: RefreshToken):
    """Refresh the access token using a valid refresh token."""
    try:
        # Validate the refresh token and get new tokens
        response = supabase.auth.refresh_session(refresh_request.refresh_token)
        
        # Extract user data
        user_data = {
            "id": response.user.id,
            "email": response.user.email,
            "username": response.user.user_metadata.get("username", "Unknown")
        }
        
        # Return the new tokens and user data
        return {
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "expires_at": int(response.session.expires_at),
            "user": user_data
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Failed to refresh token: {str(e)}")

# ======================================
# 🎮 GAME ROUTES
# ======================================
@app.post("/games", response_model=GameState)
async def create_game(game: CreateGame, user=Depends(get_current_user)):
    """Create a new game with the provided word."""
    try:
        game_id = generate_random_number_between_1000_9999()
        created_at = datetime.now().isoformat()  # Convert datetime to ISO format string

        query = """
        INSERT INTO games (id, word, guessed_letters, attempts_left, status, wrong_guesses, correct_guesses, created_by, created_at, hints)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """
        params = (game_id, game.word, [], 6, "in_progress", [], [], user.username, created_at, [])

        result = execute_query(query, params)
        if not result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create game")
        
        # Ensure created_at is a string
        result[0]['created_at'] = str(result[0]['created_at'])

        return GameState(**result[0])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/games/public", response_model=List[PublicGameResponse])
async def get_public_games(user=Depends(get_current_user)):
    """Return a list of public games that are in progress."""
    try:
        results = execute_query("SELECT * FROM games WHERE status = 'in_progress'")
        if not results:
            return []

        return [
            {
                "game_id": game["id"],
                "status": game["status"],
                "created_by": game["created_by"],
                "created_at": str(game["created_at"]),  # Convert datetime to string
                "attempts_left": game["attempts_left"]
            }
            for game in results
        ]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/games/{game_id}", response_model=PublicGameState)
async def get_game_by_id(game_id: str, user=Depends(get_current_user)):
    """Return a specific game by its ID with the word masked."""
    try:
        results = execute_query("SELECT * FROM games WHERE id = %s", (game_id,))
        if not results:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Game {game_id} not found")

        game = results[0]
        # Ensure hints is a list, default to empty list if None
        hints = game["hints"] if game["hints"] is not None else []
        
        return PublicGameState(
            id=game["id"],
            masked_word=mask_word(game["word"], game["guessed_letters"]),
            guessed_letters=game["guessed_letters"],
            attempts_left=game["attempts_left"],
            status=game["status"],
            wrong_guesses=game["wrong_guesses"],
            correct_guesses=game["correct_guesses"],
            created_by=game["created_by"],
            created_at=str(game["created_at"]),  # Convert datetime to string
            hints=hints
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/games/{game_id}/guess", response_model=PublicGameState)
async def submit_guess(game_id: str, guess: LetterGuess, user=Depends(get_current_user)):
    """Submit a letter guess for an ongoing game."""
    try:
        # Fetch the game from the database
        results = execute_query("SELECT * FROM games WHERE id = %s", (game_id,))
        if not results:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Game {game_id} not found")

        game = results[0]

        # If the game is already won or lost, prevent further guesses
        if game["status"] in ["won", "lost"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Game is already completed")

        letter = guess.letter.lower()

        # Update guessed_letters
        guessed_letters = set(game["guessed_letters"]) | {letter}

        # Check if the letter is in the word
        if letter in game["word"].lower():
            correct_guesses = set(game["correct_guesses"]) | {letter}
            wrong_guesses = set(game["wrong_guesses"])
        else:
            correct_guesses = set(game["correct_guesses"])
            wrong_guesses = set(game["wrong_guesses"]) | {letter}

        # Decrease attempts left if incorrect guess
        attempts_left = game["attempts_left"] - (1 if letter not in game["word"].lower() else 0)

        # Determine game status
        if set([c.lower() for c in game["word"] if c.isalpha()]).issubset(correct_guesses):
            status = "won"
        elif attempts_left <= 0:
            status = "lost"
        else:
            status = "in_progress"

        # Update the game state in the database
        query = """
        UPDATE games
        SET guessed_letters = %s, correct_guesses = %s, wrong_guesses = %s, attempts_left = %s, status = %s
        WHERE id = %s
        RETURNING *
        """
        params = (list(guessed_letters), list(correct_guesses), list(wrong_guesses), attempts_left, status, game_id)
        updated_game = execute_query(query, params)

        if not updated_game:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update game state")

        updated_game = updated_game[0]

        # Determine masked word or full word based on game status
        masked_word = updated_game["word"] if status == "lost" else mask_word(updated_game["word"], updated_game["guessed_letters"])

        # Return the updated game state as PublicGameState
        # Ensure hints is a list, default to empty list if None
        hints = updated_game["hints"] if updated_game["hints"] is not None else []
        
        return PublicGameState(
            id=updated_game["id"],
            masked_word=masked_word,
            guessed_letters=updated_game["guessed_letters"],
            attempts_left=updated_game["attempts_left"],
            status=updated_game["status"],
            wrong_guesses=updated_game["wrong_guesses"],
            correct_guesses=updated_game["correct_guesses"],
            created_by=updated_game["created_by"],
            created_at=str(updated_game["created_at"]),  # Convert datetime to string
            hints=hints
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/games/{game_id}/addhint", response_model=PublicGameState)
async def add_hint(game_id: str, hint: HintRequest, user=Depends(get_current_user)):
    """Add a hint to the ongoing game."""
    try:
        # Fetch the game from the database
        results = execute_query("SELECT * FROM games WHERE id = %s", (game_id,))
        if not results:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Game {game_id} not found")

        game = results[0]

        # If the game is already won or lost, prevent adding hints
        if game["status"] in ["won", "lost"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Game is already completed")

        # Update hints
        hints = game["hints"] if game["hints"] is not None else []
        hints.append(hint.hint)

        # Update the game state in the database
        query = """
        UPDATE games
        SET hints = %s
        WHERE id = %s
        RETURNING *
        """
        params = (hints, game_id)
        updated_game = execute_query(query, params)

        if not updated_game:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update game hints")

        updated_game = updated_game[0]

        # Return the updated game state as PublicGameState
        return PublicGameState(
            id=updated_game["id"],
            masked_word=mask_word(updated_game["word"], updated_game["guessed_letters"]),
            guessed_letters=updated_game["guessed_letters"],
            attempts_left=updated_game["attempts_left"],
            status=updated_game["status"],
            wrong_guesses=updated_game["wrong_guesses"],
            correct_guesses=updated_game["correct_guesses"],
            created_by=updated_game["created_by"],
            created_at=str(updated_game["created_at"]),  # Convert datetime to string
            hints=updated_game["hints"]
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# ======================================
# ⚙️ SERVER STARTUP & SHUTDOWN
# ======================================
@app.on_event("startup")
async def startup_db_client():
    """Initialize database connection on startup"""
    global db_connection
    db_connection = connect_to_db()
    if db_connection is None:
        print("Warning: Failed to establish initial database connection")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close database connection on shutdown"""
    global db_connection
    if db_connection and not db_connection.closed:
        db_connection.close()

# ======================================
# 🚀 RUN SERVER
# ======================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
