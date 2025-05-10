from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    # In a real implementation, you would call a weather API here
    weather_data = {
        "new york": "It's sunny and 75°F in New York",
        "nyc": "It's sunny and 75°F in New York",
        "san francisco": "It's foggy and 65°F in San Francisco",
        "los angeles": "It's sunny and 80°F in Los Angeles",
        "chicago": "It's windy and 60°F in Chicago",
    }

    location = location.lower()

    print(f"Received weather request for location: {location}")

    return weather_data.get(location, f"Weather data not available for {location}")


if __name__ == "__main__":
    mcp.run(transport="sse")
