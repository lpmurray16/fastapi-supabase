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
    guessed_letters: List[str] = []  # Stored as text[] in DB
    attempts_left: int
    status: str
    wrong_guesses: List[str] = []  # Stored as text[] in DB
    correct_guesses: List[str] = []  # Stored as text[] in DB
    created_by: str
    created_at: str

    @property
    def guessed_letters_list(self) -> List[str]:
        return self.guessed_letters

    @property
    def wrong_guesses_list(self) -> List[str]:
        return self.wrong_guesses

    @property
    def correct_guesses_list(self) -> List[str]:
        return self.correct_guesses

class PublicGameState(BaseModel):
    id: str
    masked_word: str
    guessed_letters: List[str] = []  # Stored as text[] in DB
    attempts_left: int
    status: str
    wrong_guesses: List[str] = []  # Stored as text[] in DB
    correct_guesses: List[str] = []  # Stored as text[] in DB
    created_by: str
    created_at: str

    @property
    def guessed_letters_list(self) -> List[str]:
        return self.guessed_letters

    @property
    def wrong_guesses_list(self) -> List[str]:
        return self.wrong_guesses

    @property
    def correct_guesses_list(self) -> List[str]:
        return self.correct_guesses

class PublicGameResponse(BaseModel):
    game_id: str
    status: str
    created_by: str
    created_at: str

class LetterGuess(BaseModel):
    letter: str

class CreateGame(BaseModel):
    word: str

# ======================================
# 🔧 UTILITY FUNCTIONS
# ======================================
def mask_word(word: str, guessed_letters: str) -> str:
    """Mask the word, replacing unguessed letters with asterisks (*) but keeping spaces visible."""
    letters_list = list(guessed_letters) if isinstance(guessed_letters, str) else guessed_letters
    return "".join([char if char.lower() in letters_list or char == " " else "*" for char in word])

def generate_random_number_between_1000_9999() -> int:
    """Generate a random number between 1000 and 9999."""
    return random.randint(1000, 9999)

# ======================================
# 🔒 AUTHENTICATION HELPERS
# ======================================
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        response = supabase.auth.get_user(token)
        # Extract user data from the response and create a custom user object with id
        user_data = response.user
        # Create a simple object with id attribute
        class UserResponse:
            def __init__(self, id):
                self.id = id
        
        return UserResponse(user_data.id)
    except Exception as e:
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
    except Exception as e:
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

# ======================================
# 🎮 GAME ROUTES
# ======================================
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
                "created_by": game["created_by"]
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
        return PublicGameState(
            id=game["id"],
            masked_word=mask_word(game["word"], game["guessed_letters"]),
            guessed_letters=game["guessed_letters"],
            attempts_left=game["attempts_left"],
            status=game["status"],

            wrong_guesses=game["wrong_guesses"],
            correct_guesses=game["correct_guesses"],
            created_by=game["created_by"],
            created_at=game["created_at"]
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/games", response_model=GameState)
async def create_game(game: CreateGame, user=Depends(get_current_user)):
    """Create a new game with the provided word."""
    try:
        game_id = generate_random_number_between_1000_9999()
        current_time = execute_query("SELECT NOW()")[0]["now"]
        
        # Create new game with default values
        query = """
        INSERT INTO games (id, word, guessed_letters, attempts_left, status, wrong_guesses, correct_guesses, created_by, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """
        
        # Set default values
        params = (
            game_id,
            game.word,
            [],  # guessed_letters
            6,   # attempts_left
            "in_progress",  # status
            [],  # wrong_guesses
            [],  # correct_guesses
            str(user.id),  # created_by
            current_time
        )
        
        result = execute_query(query, params)
        if not result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create game")
            
        return GameState(**result[0])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# ======================================
# 🔄 GAME LOGIC
# ======================================
def update_game_state(game_state: GameState, letter: str, player_id: str) -> GameState:
    # Check if the player making the guess is not the game creator
    if player_id == game_state.created_by:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Game creator cannot make guesses"
        )

    letter = letter.lower()
    
    # Update guessed_letters
    if letter not in game_state.guessed_letters:
        game_state.guessed_letters += letter
    
    if letter in game_state.word.lower():
        # Update correct_guesses
        if letter not in game_state.correct_guesses:
            game_state.correct_guesses += letter
    else:
        # Update wrong_guesses
        if letter not in game_state.wrong_guesses:
            game_state.wrong_guesses += letter
            game_state.attempts_left -= 1

    # Check if all letters in the word have been guessed
    all_word_letters = set([c.lower() for c in game_state.word if c.isalpha()])
    guessed_correct_letters = set(game_state.correct_guesses)
    
    if all_word_letters.issubset(guessed_correct_letters):
        game_state.status = "won"

    if game_state.attempts_left <= 0:
        game_state.status = "lost"

    return game_state

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
