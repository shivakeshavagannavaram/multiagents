"""Shared test fixtures."""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import networkx as nx
import pytest

from conv_gen.ingestor.registry import ToolRegistry
from conv_gen.models import (
    APIEndpoint,
    Conversation,
    JudgeScore,
    Message,
    Tool,
    ToolCall,
    ToolOutput,
    ToolParameter,
)


@pytest.fixture
def sample_parameters():
    return [
        ToolParameter(name="city", type="string", description="City name", example_value="Paris"),
        ToolParameter(name="max_price", type="integer", description="Maximum price"),
        ToolParameter(name="currency", type="string", description="Currency code", default="USD"),
    ]


@pytest.fixture
def sample_endpoint(sample_parameters):
    return APIEndpoint(
        name="search_hotels",
        url="https://api.example.com/hotels/search",
        description="Search for hotels in a given city",
        method="GET",
        required_parameters=sample_parameters[:2],
        optional_parameters=sample_parameters[2:],
    )


@pytest.fixture
def sample_tools():
    """A set of tools spanning 3 categories for testing."""
    return [
        Tool(
            tool_name="HotelFinder",
            standardized_name="hotel_finder",
            tool_description="Find and book hotels worldwide",
            category="Travel",
            api_list=[
                APIEndpoint(
                    name="search_hotels",
                    description="Search for available hotels",
                    method="GET",
                    required_parameters=[
                        ToolParameter(name="city", type="string", description="City"),
                        ToolParameter(name="check_in", type="string", description="Check-in date"),
                    ],
                    optional_parameters=[
                        ToolParameter(name="max_price", type="integer", description="Max price"),
                    ],
                    response_schema={
                        "type": "object",
                        "properties": {
                            "results": {"type": "array", "items": {"type": "object", "properties": {
                                "hotel_id": {"type": "string"},
                                "hotel_name": {"type": "string"},
                                "price": {"type": "number"},
                            }}},
                        },
                    },
                ),
                APIEndpoint(
                    name="book_hotel",
                    description="Book a hotel room",
                    method="POST",
                    required_parameters=[
                        ToolParameter(name="hotel_id", type="string", description="Hotel ID"),
                        ToolParameter(name="check_in", type="string", description="Check-in date"),
                    ],
                    response_schema={
                        "type": "object",
                        "properties": {
                            "booking_id": {"type": "string"},
                            "hotel_id": {"type": "string"},
                            "confirmation_status": {"type": "string"},
                        },
                    },
                ),
            ],
        ),
        Tool(
            tool_name="FlightSearch",
            standardized_name="flight_search",
            tool_description="Search and book flights",
            category="Travel",
            api_list=[
                APIEndpoint(
                    name="search_flights",
                    description="Search for available flights",
                    method="GET",
                    required_parameters=[
                        ToolParameter(name="origin", type="string", description="Origin city"),
                        ToolParameter(name="destination", type="string", description="Destination"),
                        ToolParameter(name="date", type="string", description="Travel date"),
                    ],
                    response_schema={
                        "type": "object",
                        "properties": {
                            "flights": {"type": "array", "items": {"type": "object", "properties": {
                                "flight_id": {"type": "string"},
                                "airline": {"type": "string"},
                                "price": {"type": "number"},
                            }}},
                        },
                    },
                ),
                APIEndpoint(
                    name="book_flight",
                    description="Book a flight ticket",
                    method="POST",
                    required_parameters=[
                        ToolParameter(name="flight_id", type="string", description="Flight ID"),
                    ],
                    response_schema={
                        "type": "object",
                        "properties": {
                            "booking_id": {"type": "string"},
                            "flight_id": {"type": "string"},
                            "ticket_status": {"type": "string"},
                        },
                    },
                ),
            ],
        ),
        Tool(
            tool_name="WeatherAPI",
            standardized_name="weather_api",
            tool_description="Get weather forecasts",
            category="Weather",
            api_list=[
                APIEndpoint(
                    name="get_forecast",
                    description="Get weather forecast for a city",
                    method="GET",
                    required_parameters=[
                        ToolParameter(name="city", type="string", description="City name"),
                    ],
                ),
            ],
        ),
        Tool(
            tool_name="RestaurantGuide",
            standardized_name="restaurant_guide",
            tool_description="Find restaurants and make reservations",
            category="Food",
            api_list=[
                APIEndpoint(
                    name="search_restaurants",
                    description="Search for restaurants in a city",
                    method="GET",
                    required_parameters=[
                        ToolParameter(name="city", type="string", description="City"),
                        ToolParameter(name="cuisine", type="string", description="Cuisine type"),
                    ],
                    response_schema={
                        "type": "object",
                        "properties": {
                            "restaurants": {"type": "array", "items": {"type": "object", "properties": {
                                "restaurant_id": {"type": "string"},
                                "restaurant_name": {"type": "string"},
                                "cuisine": {"type": "string"},
                            }}},
                        },
                    },
                ),
                APIEndpoint(
                    name="make_reservation",
                    description="Reserve a table at a restaurant",
                    method="POST",
                    required_parameters=[
                        ToolParameter(name="restaurant_id", type="string", description="Restaurant ID"),
                        ToolParameter(name="date", type="string", description="Reservation date"),
                        ToolParameter(name="party_size", type="integer", description="Number of guests"),
                    ],
                    response_schema={
                        "type": "object",
                        "properties": {
                            "reservation_id": {"type": "string"},
                            "restaurant_id": {"type": "string"},
                            "reservation_status": {"type": "string"},
                        },
                    },
                ),
            ],
        ),
        Tool(
            tool_name="CurrencyConverter",
            standardized_name="currency_converter",
            tool_description="Convert between currencies",
            category="Finance",
            api_list=[
                APIEndpoint(
                    name="convert",
                    description="Convert amount between currencies",
                    method="GET",
                    required_parameters=[
                        ToolParameter(name="from_currency", type="string", description="Source currency"),
                        ToolParameter(name="to_currency", type="string", description="Target currency"),
                        ToolParameter(name="amount", type="number", description="Amount to convert"),
                    ],
                ),
            ],
        ),
    ]


@pytest.fixture
def sample_registry(sample_tools):
    return ToolRegistry(sample_tools)


@pytest.fixture
def sample_graph(sample_registry):
    """Build a small KG from sample tools for testing."""
    from conv_gen.graph.builder import ToolGraphBuilder

    builder = ToolGraphBuilder(sample_registry)
    return builder.build()


@pytest.fixture
def sample_conversation():
    return Conversation(
        conversation_id="conv_test01",
        messages=[
            Message(role="user", content="Find me a hotel in Paris for next weekend"),
            Message(role="assistant", content="What's your budget range?"),
            Message(role="user", content="Under 200 euros per night"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        tool_name="HotelFinder",
                        api_name="search_hotels",
                        arguments={"city": "Paris", "max_price": 200},
                    )
                ],
            ),
            Message(
                role="tool",
                content=None,
                tool_outputs=[
                    ToolOutput(
                        tool_call=ToolCall(
                            tool_name="HotelFinder",
                            api_name="search_hotels",
                            arguments={"city": "Paris", "max_price": 200},
                        ),
                        response={
                            "results": [{"id": "htl_881", "name": "Hotel du Marais", "price": 175}]
                        },
                    )
                ],
            ),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        tool_name="HotelFinder",
                        api_name="book_hotel",
                        arguments={"hotel_id": "htl_881", "check_in": "2026-04-11"},
                    )
                ],
            ),
            Message(
                role="tool",
                content=None,
                tool_outputs=[
                    ToolOutput(
                        tool_call=ToolCall(
                            tool_name="HotelFinder",
                            api_name="book_hotel",
                            arguments={"hotel_id": "htl_881", "check_in": "2026-04-11"},
                        ),
                        response={"booking_id": "bk_3391", "status": "confirmed"},
                    )
                ],
            ),
            Message(role="assistant", content="I've booked Hotel du Marais for Apr 11. Confirmation: bk_3391."),
        ],
        tool_calls=[
            ToolCall(tool_name="HotelFinder", api_name="search_hotels", arguments={"city": "Paris", "max_price": 200}),
            ToolCall(tool_name="HotelFinder", api_name="book_hotel", arguments={"hotel_id": "htl_881", "check_in": "2026-04-11"}),
        ],
        tool_outputs=[
            ToolOutput(
                tool_call=ToolCall(tool_name="HotelFinder", api_name="search_hotels", arguments={"city": "Paris", "max_price": 200}),
                response={"results": [{"id": "htl_881", "name": "Hotel du Marais", "price": 175}]},
            ),
            ToolOutput(
                tool_call=ToolCall(tool_name="HotelFinder", api_name="book_hotel", arguments={"hotel_id": "htl_881", "check_in": "2026-04-11"}),
                response={"booking_id": "bk_3391", "status": "confirmed"},
            ),
        ],
        judge_scores=JudgeScore(naturalness=4.2, tool_correctness=4.8, task_completion=5.0),
        metadata={"seed": 42, "categories_list": ["Travel"]},
    )


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for Claude-based agents (user simulator, assistant)."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(type="text", text="Hello! How can I help you today?")]
    client.messages.create.return_value = response
    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for OpenAI-based agents (judge, simulator)."""
    client = MagicMock()
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = '{"naturalness": 4.0, "tool_correctness": 4.5, "task_completion": 4.0, "reasoning": "Good conversation"}'
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client
