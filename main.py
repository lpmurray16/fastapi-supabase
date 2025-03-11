import os
import json
import random
from typing import Dict, List, Optional, Union
from contextlib import contextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from random_word import RandomWords
from uuid import uuid4

# Import database connection function
from db_connect import connect_to_db


# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize database connection pool
db_connection = None

@contextmanager
def get_db_cursor():
    """Context manager for database cursor"""
    global db_connection
    
    # Create a new connection if one doesn't exist or if it's closed
    if db_connection is None or db_connection.closed:
        db_connection = connect_to_db()
        if db_connection is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to connect to database"
            )
    
    # Create a cursor
    cursor = db_connection.cursor()
    try:
        yield cursor
        db_connection.commit()
    except Exception as e:
        db_connection.rollback()
        raise e
    finally:
        cursor.close()

# Initialize FastAPI app
app = FastAPI(title="Hangman Game API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Security
security = HTTPBearer()

# Models
class UserCreate(BaseModel):
    email: str
    password: str
    username: str

class GameCreate(BaseModel):
    word: Optional[str] = None  # Word for the game (optional)
    max_players: Optional[int] = 2  # Default max players is 2

class UserLogin(BaseModel):
    email: str
    password: str

class GameState(BaseModel):
    id: str
    word: str
    guessed_letters: List[str]
    attempts_left: int
    status: str  # "in_progress", "won", "lost"
    players: List[str] = []  # List of player IDs
    current_player_index: int = 0  # Index of the current player in the players list
    max_players: int = 2  # Maximum number of players allowed
    created_at: Optional[str] = None

class GuessRequest(BaseModel):
    letter: str
    player_id: str

class JoinGameRequest(BaseModel):
    player_id: str

# Helper functions
def get_display_word(word: str, guessed_letters: List[str]) -> str:
    return "".join([letter if letter in guessed_letters else "_" for letter in word])

# Game logic functions
def update_game_state(game_state: GameState, letter: str, player_id: str) -> GameState:
    # If game is already over, return current state
    if game_state.status != "in_progress":
        return game_state
    
    # Check if it's the player's turn
    current_player = game_state.players[game_state.current_player_index]
    if current_player != player_id:
        return game_state  # Not this player's turn
        
    # Add letter to guessed letters if not already guessed
    if letter.lower() not in game_state.guessed_letters:
        game_state.guessed_letters.append(letter.lower())
        
        # Check if letter is in the word
        if letter.lower() not in game_state.word:
            game_state.attempts_left -= 1
        
        # Move to next player's turn
        game_state.current_player_index = (game_state.current_player_index + 1) % len(game_state.players)
            
    # Check win condition (all letters in word have been guessed)
    word_letters = set(game_state.word)
    guessed_correct = word_letters.intersection(set(game_state.guessed_letters))
    if len(guessed_correct) == len(word_letters):
        game_state.status = "won"
        
    # Check lose condition (no attempts left)
    if game_state.attempts_left <= 0:
        game_state.status = "lost"
        
    return game_state

# Authentication helper functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # Verify token with Supabase
        user = supabase.auth.get_user(token)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def is_valid_token(token: str) -> bool:
    """Validate a token using Supabase"""
    try:
        # Verify token with Supabase
        supabase.auth.get_user(token)
        return True
    except Exception:
        return False

# Routes
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Hangman Game API"}

# Auth routes
@app.post("/auth/register")
async def register(user: UserCreate):
    try:
        response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password,
            "options": {
                "data": {
                    "username": user.username
                }
            }
        })
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@app.post("/auth/login")
async def login(user: UserLogin):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@app.post("/auth/logout")
async def logout(user = Depends(get_current_user)):
    try:
        response = supabase.auth.sign_out()
        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

# Game routes
@app.post("/games")
async def create_game(game_data: GameCreate, user = Depends(get_current_user)):
    """Create a new game and return its ID."""
    
    # Generate a unique game ID
    new_game_id = str(random.randint(1000, 9999))
    
    # Get user ID from the token
    user_id = user.user.id
    
    # Ensure the word is set correctly
    word = game_data.word.lower() if game_data.word else RandomWords().get_random_word().lower()
    
    # Create game state
    game_state = {
        "id": new_game_id,
        "word": word,
        "guessed_letters": [],
        "attempts_left": 6,
        "status": "in_progress",
        "players": [user_id],
        "current_player_index": 0,
        "max_players": game_data.max_players
    }
    
    # Store game in database using direct connection
    try:
        # Convert Python list to PostgreSQL array format
        players_array = json.dumps(game_state["players"])
        guessed_letters_array = json.dumps(game_state["guessed_letters"])
        
        query = """
        INSERT INTO games (id, word, guessed_letters, attempts_left, status, players, current_player_index, max_players)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        execute_query(
            query, 
            (new_game_id, word, guessed_letters_array, 6, "in_progress", players_array, 0, game_data.max_players)
        )
        
        # We can still use Supabase for realtime features
        supabase.channel(f"game-{new_game_id}").subscribe()
        
        return {"game_id": new_game_id, "message": "Game created successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create game: {str(e)}"
        )

@app.get("/games/{game_id}")
async def get_game(game_id: str, user = Depends(get_current_user)):
    """Get the current state of a game."""
    try:
        # Use direct database connection instead of Supabase
        query = "SELECT * FROM games WHERE id = %s"
        results = execute_query(query, (game_id,))
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        game_state = results[0]
        
        # Hide the actual word in the response
        display_data = {
            "id": game_state["id"],
            "display_word": get_display_word(game_state["word"], game_state["guessed_letters"]),
            "guessed_letters": game_state["guessed_letters"],
            "attempts_left": game_state["attempts_left"],
            "status": game_state["status"],
            "players": game_state["players"],
            "current_player": game_state["players"][game_state["current_player_index"]],
            "max_players": game_state["max_players"]
        }
        
        # If game is over, reveal the word
        if game_state["status"] != "in_progress":
            display_data["word"] = game_state["word"]
        
        return display_data
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get game: {str(e)}"
        )

@app.post("/games/{game_id}/join")
async def join_game(game_id: str, join_request: JoinGameRequest, user = Depends(get_current_user)):
    """Join an existing game."""
    try:
        # Get the current game state
        response = supabase.table("games").select("*").eq("id", game_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        game_state = response.data[0]
        
        # Check if game is full
        if len(game_state["players"]) >= game_state["max_players"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Game is full"
            )
        
        # Check if player is already in the game
        if join_request.player_id in game_state["players"]:
            return {"message": "Already in the game"}
        
        # Add player to the game
        game_state["players"].append(join_request.player_id)
        
        # Update the game in Supabase
        supabase.table("games").update({"players": game_state["players"]}).eq("id", game_id).execute()
        
        return {"message": "Successfully joined the game"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to join game: {str(e)}"
        )

@app.post("/games/{game_id}/guess")
async def make_guess(game_id: str, guess_request: GuessRequest, user = Depends(get_current_user)):
    """Make a guess in the game."""
    try:
        # Get the current game state
        response = supabase.table("games").select("*").eq("id", game_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        game_state_dict = response.data[0]
        
        # Convert to GameState model
        game_state = GameState(**game_state_dict)
        
        # Update game state with the guess
        updated_game_state = update_game_state(game_state, guess_request.letter, guess_request.player_id)
        
        # Update the game in Supabase
        supabase.table("games").update({
            "guessed_letters": updated_game_state.guessed_letters,
            "attempts_left": updated_game_state.attempts_left,
            "status": updated_game_state.status,
            "current_player_index": updated_game_state.current_player_index
        }).eq("id", game_id).execute()
        
        # Prepare response data
        display_data = {
            "display_word": get_display_word(updated_game_state.word, updated_game_state.guessed_letters),
            "guessed_letters": updated_game_state.guessed_letters,
            "attempts_left": updated_game_state.attempts_left,
            "status": updated_game_state.status,
            "current_player": updated_game_state.players[updated_game_state.current_player_index]
        }
        
        # If game is over, reveal the word
        if updated_game_state.status != "in_progress":
            display_data["word"] = updated_game_state.word
        
        return display_data
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process guess: {str(e)}"
        )

@app.delete("/games/{game_id}")
async def delete_game(game_id: str, user = Depends(get_current_user)):
    """Delete a game and optionally create a new one."""
    try:
        # Get the current game state to verify it exists
        response = supabase.table("games").select("*").eq("id", game_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        # Delete the game from Supabase
        supabase.table("games").delete().eq("id", game_id).execute()
        
        return {"message": "Game deleted successfully"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete game: {str(e)}"
        )

@app.post("/games/{game_id}/restart")
async def restart_game(game_id: str, game_data: GameCreate = None, user = Depends(get_current_user)):
    """Delete the current game and create a new one with the same ID."""
    try:
        # Get the current game state to verify it exists and get players
        response = supabase.table("games").select("*").eq("id", game_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        old_game = response.data[0]
        players = old_game["players"]
        
        # Delete the old game
        supabase.table("games").delete().eq("id", game_id).execute()
        
        # Create a new game with the same ID
        word = game_data.word.lower() if game_data and game_data.word else RandomWords().get_random_word().lower()
        max_players = game_data.max_players if game_data else 2
        
        new_game_state = {
            "id": game_id,
            "word": word,
            "guessed_letters": [],
            "attempts_left": 6,
            "status": "in_progress",
            "players": players,
            "current_player_index": 0,
            "max_players": max_players
        }
        
        # Store new game in Supabase
        supabase.table("games").insert(new_game_state).execute()
        
        return {"message": "Game restarted successfully", "game_id": game_id}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart game: {str(e)}"
        )

@app.get("/games/public")
async def get_public_games(user = Depends(get_current_user)):
    """Return a list of public games that are in progress and not full."""
    try:
        response = supabase.table("games").select("*").eq("status", "in_progress").execute()
        
        public_games = []
        for game in response.data:
            if len(game["players"]) < game["max_players"]:
                public_games.append({
                    "game_id": game["id"],
                    "players": game["players"],
                    "max_players": game["max_players"],
                    "status": game["status"]
                })
        
        return public_games
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get public games: {str(e)}"
        )

# Database operations
def execute_query(query, params=None):
    """Execute a database query and return results"""
    with get_db_cursor() as cursor:
        cursor.execute(query, params or ())
        try:
            results = cursor.fetchall()
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            # Convert to list of dictionaries
            return [dict(zip(column_names, row)) for row in results]
        except:
            # For queries that don't return results (INSERT, UPDATE, DELETE)
            return None

# Event handlers for application startup and shutdown
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
    if db_connection is not None and not db_connection.closed:
        db_connection.close()
        print("Database connection closed")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)