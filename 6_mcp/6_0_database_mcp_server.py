import asyncio
import sqlite3
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Initialize the MCP server
server = Server("sqlite-database")

# Create a simple SQLite database
def init_database():
    conn = sqlite3.connect('simple_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="add_user",
            description="Add a new user to the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's name"},
                    "email": {"type": "string", "description": "User's email"}
                },
                "required": ["name", "email"]
            }
        ),
        Tool(
            name="list_users",
            description="List all users in the database",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "add_user":
        conn = sqlite3.connect('simple_db.sqlite')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (arguments["name"], arguments["email"])
        )
        conn.commit()
        conn.close()
        return [TextContent(
            type="text",
            text=f"Added user: {arguments['name']} ({arguments['email']})"
        )]
    
    elif name == "list_users":
        conn = sqlite3.connect('simple_db.sqlite')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        conn.close()
        
        if not users:
            return [TextContent(type="text", text="No users found")]
        
        result = "Users:\n" + "\n".join(
            f"ID: {user[0]}, Name: {user[1]}, Email: {user[2]}" 
            for user in users
        )
        return [TextContent(type="text", text=result)]
    
    return [TextContent(type="text", text="Unknown tool")]

async def main():
    init_database()
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())