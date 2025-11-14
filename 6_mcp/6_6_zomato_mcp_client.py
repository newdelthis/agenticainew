import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="npx",
        args=["mcp-remote", "https://mcp-server.zomato.com/mcp"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            tool_name = "get_restaurants_for_keyword"

            # Find schema (for debugging, optional)
            tool = next((t for t in tools_response.tools if t.name == tool_name), None)
            print(f"\n=== Schema for '{tool_name}' ===")
            print(tool.inputSchema)

            payload = {
            "user_location": {
                "latitude": "18.5204",
                "longitude": "73.8567",
                "cell_id": "1f54d04d57",  # Pune
                "place_id": "ChIJARFGZy6_wjsRQ-Oenb9DjYI",  # actual Pune place_id from Google
                "place_type": "GOOGLE_PLACE",
                "short_name": "Pune",
                "full_name": "Pune, Maharashtra, India",
                "delivery_subzone_id": "dsz_pune_01",
                "cell_details": {
                    "country_id": 1,
                    "city_id": 6,
                    "dsz_id": 101,
                    "cell_id": "1f54d04d57"
                }
            },
            "keyword": "Pizza",
            "page_size": 5
        }
            
            result = await session.call_tool(tool_name, payload)
            print("\n=== Result ===")
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
