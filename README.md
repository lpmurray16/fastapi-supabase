# Hangman Game API

A FastAPI-based backend for a multiplayer Hangman game with Supabase authentication.

## Features

- User authentication (register, login, logout) using Supabase
- Real-time multiplayer gameplay using WebSockets
- Custom word support or random word generation
- Game state management

## Technologies Used

- FastAPI: Modern, fast web framework for building APIs
- Supabase: Open source Firebase alternative for authentication and database
- WebSockets: For real-time communication
- Pydantic: Data validation and settings management
- Uvicorn: ASGI server for running the application

## Setup

### Prerequisites

- Python 3.7+
- Supabase account and project

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/fastapi-supabase-hangman.git
cd fastapi-supabase-hangman
```

2. Create a virtual environment and activate it

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Supabase credentials

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Running the Application

```bash
python main.py
```

Or using Uvicorn directly:

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

### Authentication

- `POST /auth/register`: Register a new user
- `POST /auth/login`: Login with email and password
- `POST /auth/logout`: Logout (requires authentication)

### WebSocket

- `WebSocket /ws/game/{game_id}`: Connect to a game session
  - Optional query parameter: `word` to set a custom word

## WebSocket Communication

### Client to Server

```json
{"type": "guess", "letter": "a"}
```

### Server to Client

```json
{"type": "game_state", "data": {"word": "*****", "display_word": "_a__", "guessed_letters": ["a", "e"], "attempts_left": 5, "status": "in_progress"}}
```

```json
{"type": "game_over", "data": {"word": "apple", "status": "won"}}
```

## License

MIT