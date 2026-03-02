from fastapi import FastAPI
from langserve import add_routes
from main import chatbot_chain

app = FastAPI()

# Create chain
#joke_chain = get_joke_chain()

# Expose as API at /joke
#add_routes(app, joke_chain, path="/joke")
add_routes(app, chatbot_chain, path="/chatbot")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)