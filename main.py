import os
import json
import random
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from random_word import RandomWords

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

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

class UserLogin(BaseModel):
    email: str
    password: str

class GameState(BaseModel):
    word: str
    guessed_letters: List[str]
    attempts_left: int
    status: str  # "in_progress", "won", "lost"

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.game_states: Dict[str, GameState] = {}
        self.word_generator = RandomWords()

    async def connect(self, websocket: WebSocket, game_id: str, custom_word: Optional[str] = None):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
            # Initialize game state if it's a new game
            if game_id not in self.game_states:
                # Use custom word if provided, otherwise generate a random word
                if custom_word:
                    word = custom_word
                else:
                    word = self.word_generator.get_random_word()
                self.game_states[game_id] = GameState(
                    word=word.lower(),
                    guessed_letters=[],
                    attempts_left=6,
                    status="in_progress"
                )
        self.active_connections[game_id].append(websocket)

    def disconnect(self, websocket: WebSocket, game_id: str):
        self.active_connections[game_id].remove(websocket)
        if not self.active_connections[game_id]:
            # Clean up if no connections left for this game
            del self.active_connections[game_id]
            if game_id in self.game_states:
                del self.game_states[game_id]

    async def broadcast(self, message: dict, game_id: str):
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id]:
                await connection.send_json(message)

    def get_game_state(self, game_id: str) -> Optional[GameState]:
        return self.game_states.get(game_id)

    def update_game_state(self, game_id: str, letter: str) -> GameState:
        game_state = self.game_states[game_id]
        
        # If game is already over, return current state
        if game_state.status != "in_progress":
            return game_state
            
        # Add letter to guessed letters if not already guessed
        if letter.lower() not in game_state.guessed_letters:
            game_state.guessed_letters.append(letter.lower())
            
            # Check if letter is in the word
            if letter.lower() not in game_state.word:
                game_state.attempts_left -= 1
                
        # Check win condition (all letters in word have been guessed)
        word_letters = set(game_state.word)
        guessed_correct = word_letters.intersection(set(game_state.guessed_letters))
        if len(guessed_correct) == len(word_letters):
            game_state.status = "won"
            
        # Check lose condition (no attempts left)
        if game_state.attempts_left <= 0:
            game_state.status = "lost"
            
        return game_state

# Initialize connection manager
manager = ConnectionManager()

# Authentication helper function
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

# Game WebSocket route
@app.websocket("/ws/game/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    # Get query parameters
    query_params = dict(websocket.query_params)
    custom_word = query_params.get("word", None)
    
    await manager.connect(websocket, game_id, custom_word)
    try:
        # Send initial game state
        game_state = manager.get_game_state(game_id)
        await websocket.send_json({
            "type": "game_state",
            "data": {
                "word": "*" * len(game_state.word),  # Hide the actual word
                "display_word": get_display_word(game_state.word, game_state.guessed_letters),
                "guessed_letters": game_state.guessed_letters,
                "attempts_left": game_state.attempts_left,
                "status": game_state.status
            }
        })
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "guess":
                letter = message["letter"]
                
                # Update game state
                game_state = manager.update_game_state(game_id, letter)
                
                # Broadcast updated game state to all players
                await manager.broadcast({
                    "type": "game_state",
                    "data": {
                        "word": "*" * len(game_state.word),  # Hide the actual word
                        "display_word": get_display_word(game_state.word, game_state.guessed_letters),
                        "guessed_letters": game_state.guessed_letters,
                        "attempts_left": game_state.attempts_left,
                        "status": game_state.status
                    }
                }, game_id)
                
                # If game is over, reveal the word
                if game_state.status != "in_progress":
                    await manager.broadcast({
                        "type": "game_over",
                        "data": {
                            "word": game_state.word,
                            "status": game_state.status
                        }
                    }, game_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket, game_id)
        await manager.broadcast({"type": "info", "data": "A player has left the game"}, game_id)
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close()

# Helper function to get display word with guessed letters revealed
def get_display_word(word: str, guessed_letters: List[str]) -> str:
    return "".join([letter if letter in guessed_letters else "_" for letter in word])

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)