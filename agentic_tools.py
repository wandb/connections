import asyncio
import aiosqlite
import argparse
import logging
import pprint
import hashlib
import itertools
import json
import os
import random
import uuid
import sqlite3
import tempfile

from dataclasses import dataclass, field
from typing import List, TypedDict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import weave

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# specify the version of the agent
__version__ = "0.1.0"

pp = pprint.PrettyPrinter(indent=4)

db_lock = asyncio.Lock()

MAX_ERRORS = 4
RETRY_LIMIT = 8

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# define the state of the puzzle
class PuzzleState(TypedDict):
    puzzle_status: str = ""
    tool_status: str = ""
    current_tool: str = ""
    workflow_instructions: Optional[str] = None
    vocabulary_db_fp: Optional[str] = None
    tool_to_use: str = ""
    words_remaining: List[str] = []
    invalid_connections: Optional[List[Tuple[str, List[str]]]] = []
    recommended_words: List[str] = []
    recommended_connection: str = ""
    recommended_correct: bool = False
    recommendation_answer_status: Optional[str] = None
    recommendation_correct_groups: Optional[List[List[str]]] = []
    puzzle_checker_response: str = ""
    mistake_count: int = 0
    llm_retry_count: int = 0
    found_count: int = 0
    recommendation_count: int = 0
    llm_temperature: float = 1.0
    puzzle_source_type: Optional[str] = None
    puzzle_source_fp: Optional[str] = None


@weave.op()
def chat_with_llm(prompt, model="gpt-4o", temperature=0.7, max_tokens=4096):
    """
    Interact with a language model (LLM) using a given prompt.

    Args:
        prompt (str): The input text to be sent to the language model.
        model (str, optional): The model to use for generating responses. Defaults to "gpt-4o".
        temperature (float, optional): The sampling temperature to use. Higher values mean more random completions. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 4096.

    Returns:
        dict: The response from the language model in JSON format.
    """

    # Initialize the OpenAI LLM with your API key and specify the GPT-4o model
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    result = llm.invoke(prompt)

    return result


@weave.op()
async def apply_recommendation(state: PuzzleState) -> PuzzleState:
    """
    Apply the recommendation to the current puzzle state and update the state accordingly.

    Args:
        state (PuzzleState): The current state of the puzzle.

    Returns:
        PuzzleState: The updated state of the puzzle after applying the recommendation.

    The function performs the following steps:
    1. Logs the entry into the function and the current state.
    2. Increments the recommendation count.
    3. Retrieves the user response from the puzzle checker.
    4. Processes the result of the user response:
        - If the response is "correct":
            - Logs and prints the correct recommendation.
            - If the current tool is "embedvec_recommender", removes the accepted words from the vocabulary database.
            - Removes the recommended words from the remaining words.
            - Updates the state to reflect the correct recommendation.
        - If the response is "one_away" or "incorrect":
            - Logs and prints the incorrect recommendation.
            - Updates the state to reflect the incorrect recommendation.
            - If the mistake count is less than the maximum allowed errors, performs additional analysis for "one_away" responses.
            - If the mistake count exceeds the maximum allowed errors, resets the recommendation state.
    5. Checks if the puzzle is completed or if the maximum allowed errors are reached, and updates the state accordingly.
    6. Logs the exit from the function and the updated state.

    Note:
        - The function uses asynchronous database operations with aiosqlite.
        - The function handles different tools for recommendations, such as "embedvec_recommender" and "llm_recommender".
    """
    logger.info("Entering apply_recommendation:")
    logger.debug(f"\nEntering apply_recommendation State: {pp.pformat(state)}")

    state["recommendation_count"] += 1

    # get user response from puzzle checkter
    checker_response = state["puzzle_checker_response"]

    # process result of user response
    if checker_response == "correct":
        print(f"Recommendation {sorted(state['recommended_words'])} is correct")

        # for embedvec_recommender, remove the words from the vocabulary database
        if state["current_tool"] == "embedvec_recommender":
            async with aiosqlite.connect(state["vocabulary_db_fp"]) as conn:
                async with db_lock:
                    # remove accepted words from vocabulary.db
                    # for each word in recommended_words, remove the word from the vocabulary table
                    for word in state["recommended_words"]:
                        sql_query = f"DELETE FROM vocabulary WHERE word = '{word}'"
                        await conn.execute(sql_query)
                    await conn.commit()
                    # await conn.close()

        # remove the words from words_remaining
        state["words_remaining"] = [
            word
            for word in state["words_remaining"]
            if word not in state["recommended_words"]
        ]
        state["recommended_correct"] = True
        state["found_count"] += 1
        state["recommendation_correct_groups"].append(
            {
                "words": state["recommended_words"],
                "reason": state["recommended_connection"],
            }
        )

    elif checker_response in ["one_away", "incorrect"]:
        invalid_group = state["recommended_words"]
        invalid_group_id = compute_group_id(invalid_group)
        state["invalid_connections"].append((invalid_group_id, invalid_group))
        state["recommended_correct"] = False
        state["mistake_count"] += 1

        if state["mistake_count"] < MAX_ERRORS:
            match checker_response:
                case "one_away":
                    print(
                        f"Recommendation {sorted(state['recommended_words'])} is incorrect, one away from correct"
                    )
                    one_away_group_recommendation = None

                    if state["current_tool"] == "embedvec_recommender":
                        print(
                            "Changing the recommender from 'embedvec_recommender' to 'llm_recommender'"
                        )
                        state["current_tool"] = "llm_recommender"
                    else:

                        # perform one-away analysis
                        one_away_group_recommendation = one_away_analyzer(
                            state, invalid_group, state["words_remaining"]
                        )

                        # check if one_away_group_recommendation is a prior mistake
                        if one_away_group_recommendation:
                            one_away_group_id = compute_group_id(
                                one_away_group_recommendation.words
                            )
                            if one_away_group_id in set(
                                x[0] for x in state["invalid_connections"]
                            ):
                                print(
                                    f"one_away_group_recommendation is a prior mistake"
                                )
                                one_away_group_recommendation = None
                            else:
                                print(
                                    f"one_away_group_recommendation is a new recommendation"
                                )

                case "incorrect":
                    print(
                        f"Recommendation {sorted(state['recommended_words'])} is incorrect"
                    )
                    if state["current_tool"] == "embedvec_recommender":
                        print(
                            "Changing the recommender from 'embedvec_recommender' to 'llm_recommender'"
                        )
                        state["current_tool"] = "llm_recommender"

        else:
            state["recommended_words"] = []
            state["recommended_connection"] = ""
            state["recommended_correct"] = False

    if len(state["words_remaining"]) == 0 or state["mistake_count"] >= MAX_ERRORS:
        if state["mistake_count"] >= MAX_ERRORS:
            logger.info("FAILED TO SOLVE THE CONNECTION PUZZLE TOO MANY MISTAKES!!!")
            print("FAILED TO SOLVE THE CONNECTION PUZZLE TOO MANY MISTAKES!!!")

        else:
            logger.info("SOLVED THE CONNECTION PUZZLE!!!")
            print("SOLVED THE CONNECTION PUZZLE!!!")

        state["tool_status"] = "puzzle_completed"

    elif checker_response == "one_away":
        if one_away_group_recommendation:
            print(f"using one_away_group_recommendation")
            state["recommended_words"] = one_away_group_recommendation.words
            state["recommended_connection"] = (
                one_away_group_recommendation.connection_description
            )
            state["tool_status"] = "have_recommendation"

        else:
            print(f"no one_away_group_recommendation, let llm_recommender try again")
            state["recommended_words"] = []
            state["recommended_connection"] = ""
            state["tool_status"] = "next_recommendation"

    else:
        logger.info("Going to next get_recommendation")
        state["tool_status"] = "next_recommendation"

    logger.info("Exiting apply_recommendation:")
    logger.debug(f"\nExiting apply_recommendation State: {pp.pformat(state)}")

    return state


SYSTEM_MESSAGE_LLM = SystemMessage(
    """
    You are a helpful assistant in solving the New York Times Connection Puzzle.

    The New York Times Connection Puzzle involves identifying groups of four related items from a grid of 16 words. Each word can belong to only one group, and there are generally 4 groups to identify. Your task is to examine the provided words, identify the possible groups based on thematic connections, and then suggest the groups one by one.

    # Steps

    1. **Review the candidate words**: Look at thewords provided in the candidate list carefully.
    2. **Identify Themes**: Notice any apparent themes or categories (e.g., types of animals, names of colors, etc.).
    3. **Group Words**: Attempt to form groups of four words that share a common theme.
    4. **Avoid invalid groups**: Do not include word groups that are known to be invalid.
    5. **Verify Groups**: Ensure that each word belongs to only one group. If a word seems to fit into multiple categories, decide on the best fit based on the remaining options.
    6. **Order the groups**: Order your answers in terms of your confidence level, high confidence first.
    7. **Solution output**: Generate only a json response as shown in the **Output Format** section.

    # Output Format

    Provide the solution with the identified groups and their themes in a structured format. Each group should be output as a JSON list object.  Each list item is dictionary with keys "words" list of the connected words and "connection" describing the connection among the words.

    ```json
    [
    {"words": ["Word1", "Word2", "Word3", "Word4"], "connection": "..."},
    {"words": ["Word5", "Word6", "Word7", "Word8"], "connection": "..."},
    {"words": ["Word9", "Word10", "Word11", "Word12"], "connection": "..."},
    {"words": ["Word13", "Word14", "Word15", "Word16"], "connection": "..."}
    ]
    ```

    # Examples

    **Example:**

    - **Input:** ["prime", "dud", "shot", "card", "flop", "turn", "charge", "rainforest", "time", "miss", "plastic", "kindle", "chance", "river", "bust", "credit"]
    
    - **Output:**
    [
    {"words": [ "bust", "dud", "flop", "mist"], "connection": "clunker"},
    {"words": ["chance", "shot", "time", "turn"], "connection": "opportunity"},
    {"words": ["card", "charge", "credit", "plastic"], "connection": "Non-Cash Way to Pay"},
    {"words": ["kindle", "prime", "rainforest", "river"], "connection": "Amazon ___"}
    ]

    No other text.

    # Notes

    - Ensure all thematic connections make logical sense.
    - Consider edge cases where a word could potentially fit into more than one category.
    - Focus on clear and accurate thematic grouping to aid in solving the puzzle efficiently.
    """
)


@weave.op()
def ask_llm_for_solution(prompt, model="gpt-4o", temperature=1.0, max_tokens=4096):
    """
    Asks a language model (LLM) for a solution based on the provided prompt.

    Args:
        prompt (str): The input prompt to be sent to the LLM.
        model (str, optional): The model to be used for generating the response. Defaults to "gpt-4o".
        temperature (float, optional): The sampling temperature to use. Higher values mean more random completions. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 4096.

    Returns:
        response: The response from the LLM.
    """
    logger.info("Entering ask_llm_for_solution")
    logger.debug(f"Entering ask_llm_for_solution Prompt: {prompt.content}")

    # Create a prompt by concatenating the system and human messages
    conversation = [SYSTEM_MESSAGE_LLM, prompt]

    # Invoke the LLM
    response = chat_with_llm(
        conversation,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info("Exiting ask_llm_for_solution")
    logger.debug(f"exiting ask_llm_for_solution response {response.content}")

    return response


KEY_PUZZLE_STATE_FIELDS = ["puzzle_status", "tool_status", "current_tool"]


WORKFLOW_SPECIFICATION = """
    **Instructions**

    use "setup_puzzle" tool to initialize the puzzle if the "puzzle_status" is not initialized.

    if "tool_status" is "puzzle_completed" then use "END" tool.

    Use the table to select the appropriate tool.

    |current_tool| tool_status | tool |
    | --- | --- | --- |
    |setup_puzzle| initialized | get_embedvec_recommendation |
    |embedvec_recommender| next_recommendation | get_embedvec_recommendation |
    |embedvec_recommender| have_recommendation | apply_recommendation |
    |llm_recommender| next_recommendation | get_llm_recommendation |
    |llm_recommender| have_recommendation | apply_recommendation |

    If no tool is selected, use "ABORT" tool.
"""


@weave.op()
def run_planner(state: PuzzleState) -> PuzzleState:
    """
    Executes the planning logic for the given puzzle state.

    This function logs the entry and exit points, processes the workflow instructions,
    converts the state to a JSON string, wraps it in a human message, and queries the
    language model for the next action. The resulting action is then used to update
    the state.

    Args:
        state (PuzzleState): The current state of the puzzle.

    Returns:
        PuzzleState: The updated state after processing the next action.
    """
    logger.info("Entering run_planner:")
    logger.debug(f"\nEntering run_planner State: {pp.pformat(state)}")

    if state["workflow_instructions"] is None:
        state["workflow_instructions"] = WORKFLOW_SPECIFICATION

        logger.debug(f"Workflow Specification: {state['workflow_instructions']}")

    # workflow instructions
    instructions = HumanMessage(WORKFLOW_SPECIFICATION)
    logger.debug(f"\nWorkflow instructions:\n{instructions.content}")

    # convert state to json string
    relevant_state = {k: state[k] for k in KEY_PUZZLE_STATE_FIELDS}
    puzzle_state = "\npuzzle state:\n" + json.dumps(relevant_state)

    # wrap the state in a human message
    puzzle_state = HumanMessage(puzzle_state)
    logger.info(f"\nState for lmm: {puzzle_state.content}")

    # get next action from llm
    next_action = ask_llm_for_next_step(
        instructions, puzzle_state, model="gpt-3.5-turbo", temperature=0
    )

    logger.info(f"\nNext action from llm: {next_action.content}")

    state["tool_to_use"] = json.loads(next_action.content)["tool"]

    logger.info("Exiting run_planner:")
    logger.debug(f"\nExiting run_planner State: {pp.pformat(state)}")
    return state


@weave.op()
def determine_next_action(state: PuzzleState) -> str:
    """
    Determines the next action to take based on the given puzzle state.

    Args:
        state (PuzzleState): The current state of the puzzle, which includes the tool to use.

    Returns:
        str: The next action to take, which could be a specific tool or an end signal.

    Raises:
        ValueError: If the tool to use is "ABORT", indicating that the process should be aborted.

    Notes:
        - Logs the entry into the function and the current state for debugging purposes.
        - If the tool to use is "END", it returns a predefined END signal.
        - Otherwise, it returns the tool specified in the state.
    """
    logger.info("Entering determine_next_action:")
    logger.debug(f"\nEntering determine_next_action State: {pp.pformat(state)}")

    tool_to_use = state["tool_to_use"]

    if tool_to_use == "ABORT":
        raise ValueError("LLM returned abort")
    elif tool_to_use == "END":
        return END
    else:
        return tool_to_use


HUMAN_MESSAGE_BASE = """
    From the following candidate list of words identify a group of four words that are connected by a common word association, theme, concept, or category, and describe the connection.      
    """


@weave.op()
def get_llm_recommendation(state: PuzzleState) -> PuzzleState:
    """
    Generates a recommendation for the next move in a puzzle using a language model (LLM).

    Args:
        state (PuzzleState): The current state of the puzzle, including information about found words, mistakes, and remaining words.

    Returns:
        PuzzleState: The updated state of the puzzle with the recommended words and connection.

    The function performs the following steps:
    1. Logs the entry into the function and the current state.
    2. Sets the current tool to "llm_recommender" and prints the current tool status.
    3. Constructs a prompt for the LLM based on the remaining words.
    4. Attempts to get a valid recommendation from the LLM, retrying up to a defined limit.
    5. If a valid recommendation is obtained, updates the state with the recommended words and connection.
    6. If no valid recommendation is obtained after the retry limit, switches to manual recommendation mode.
    7. Logs the exit from the function and the updated state.

    Note:
        - The function uses a retry mechanism to handle invalid recommendations.
        - The state is updated with the recommended words and connection if a valid recommendation is obtained.
        - If the retry limit is exceeded, the tool status is changed to "manual_recommendation".
    """
    logger.info("Entering get_recommendation")
    logger.debug(f"Entering get_recommendation State: {pp.pformat(state)}")

    state["current_tool"] = "llm_recommender"
    print(f"\nENTERED {state['current_tool'].upper()}")
    print(
        f"found count: {state['found_count']}, mistake_count: {state['mistake_count']}"
    )

    # build prompt for llm
    prompt = HUMAN_MESSAGE_BASE

    attempt_count = 0
    while True:
        attempt_count += 1
        if attempt_count > RETRY_LIMIT:
            break
        print(f"attempt_count: {attempt_count}")
        prompt = HUMAN_MESSAGE_BASE
        # scramble the remaining words for more robust group selection
        if np.random.uniform() < 0.5:
            random.shuffle(state["words_remaining"])
        else:
            state["words_remaining"].reverse()
        print(f"words_remaining: {state['words_remaining']}")
        prompt += f"candidate list: {', '.join(state['words_remaining'])}\n"

        prompt = HumanMessage(prompt)

        logger.info(f"\nPrompt for llm: {prompt.content}")

        # get recommendation from llm
        llm_response = ask_llm_for_solution(
            prompt, temperature=state["llm_temperature"]
        )

        llm_response_json = json.loads(llm_response.content)
        if isinstance(llm_response_json, list):
            logger.debug(f"\nLLM response is list")
            recommended_words = llm_response_json[0]["words"]
            recommended_connection = llm_response_json[0]["connection"]
        else:
            logger.debug(f"\nLLM response is dict")
            recommended_words = llm_response_json["words"]
            recommended_connection = llm_response_json["connection"]

        if compute_group_id(recommended_words) not in set(
            x[0] for x in state["invalid_connections"]
        ):
            break
        else:
            print(
                f"\nrepeat invalid group detected: group_id {compute_group_id(recommended_words)}, recommendation: {sorted(recommended_words)}"
            )

    state["recommended_words"] = sorted(recommended_words)
    state["recommended_connection"] = recommended_connection

    if attempt_count <= RETRY_LIMIT:
        state["tool_status"] = "have_recommendation"
    else:
        print(f"Failed to get a valid recommendation after {RETRY_LIMIT} attempts")
        print("Changing to manual_recommender, last attempt to solve the puzzle")
        print(
            f"last recommendation: {state['recommended_words']} with {state['recommended_connection']}"
        )
        state["tool_status"] = "manual_recommendation"

    logger.info("Exiting get_recommendation")
    logger.debug(f"Exiting get_recommendation State: {pp.pformat(state)}")

    return state


@weave.op()
async def get_embedvec_recommendation(state: PuzzleState) -> PuzzleState:
    """
    Asynchronously generates a recommendation for embedding vectors based on the given puzzle state.

    This function connects to a vocabulary database, retrieves a list of words with their embeddings,
    and processes them to generate a list of candidate words. It then validates the top candidates
    using a language model and updates the puzzle state with the recommended words and connection.

    Args:
        state (PuzzleState): The current state of the puzzle, containing necessary information such as
                             database file path, found count, and mistake count.

    Returns:
        PuzzleState: The updated puzzle state with the recommended words and connection.

    Raises:
        Exception: If there is an issue with database connection or data processing.

    Logging:
        Logs entry and exit points, as well as debug information about the state and recommendations.
    """
    logger.info("Entering get_embedvec_recommendation")
    logger.debug(f"Entering get_embedvec_recommendation State: {pp.pformat(state)}")

    state["current_tool"] = "embedvec_recommender"
    print(f"\nENTERED {state['current_tool'].upper()}")
    print(
        f"found count: {state['found_count']}, mistake_count: {state['mistake_count']}"
    )

    async with aiosqlite.connect(state["vocabulary_db_fp"]) as conn:
        async with db_lock:
            # get candidate list of words from database
            sql_query = "SELECT word, definition, embedding FROM vocabulary"
            async with conn.execute(sql_query) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                df = pd.DataFrame(rows, columns=columns)
                # await conn.close()

    # convert embedding string representation to numpy array
    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x)))

    # get candidate list of words based on embedding vectors
    candidate_list = get_candidate_words(df)
    print(f"candidate_lists size: {len(candidate_list)}")

    # validate the top 5 candidate list with LLM
    list_to_validate = "\n".join([str(x) for x in candidate_list[:5]])
    recommended_group = choose_embedvec_item(list_to_validate)
    logger.info(f"Recommended group: {recommended_group}")

    state["recommended_words"] = recommended_group["candidate_group"]
    state["recommended_connection"] = recommended_group["explanation"]
    state["tool_status"] = "have_recommendation"

    # build prompt for llm

    logger.info("Exiting get_embedvec_recommendation")
    logger.debug(f"Exiting get_embedvec_recommendation State: {pp.pformat(state)}")

    return state


def configure_logging(log_level):
    """
    Configures the logging settings for the application.

    Args:
        log_level (str): The logging level to set. This should be a string
                         representing the logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

    Raises:
        ValueError: If the provided log_level is not a valid logging level.

    The logging configuration includes:
        - Setting the logging level.
        - Defining the log format.
        - Logging to a file named 'app.log'.
        - Optionally, logging to the console (commented out by default).
    """
    # get numeric value of log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure the logging settings
    logging.basicConfig(
        level=numeric_level,  # Set the logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define the log format
        handlers=[
            logging.FileHandler("app.log"),  # Log to a file
            # logging.StreamHandler(),  # Optional: Log to the console as well
        ],
    )


def compute_group_id(word_group: list) -> str:
    """
    Computes a unique group ID for a list of words.

    This function takes a list of words, sorts them, concatenates them into a single string,
    and then generates an MD5 hash of that string. The resulting hash is returned as the group ID.

    Args:
        word_group (list): A list of words for which to compute the group ID.

    Returns:
        str: The MD5 hash of the concatenated and sorted words, representing the group ID.
    """
    return hashlib.md5("".join(sorted(word_group)).encode()).hexdigest()


# used by the embedvec tool to store the candidate groups
@dataclass
class ConnectionGroup:
    """
    A class to represent a group of connections with associated metrics and identifiers.

    Attributes:
        group_metric (float): Average cosine similarity of all combinations of words in the group.
        root_word (str): Root word of the group.
        candidate_pairs (list): List of candidate words with their definitions.
        group_id (Optional[str]): Checksum identifier for the group.

    Methods:
        add_entry(word, connection):
            Adds a word and its connection to the candidate pairs if the group is not full.
            Raises ValueError if the group is full.

        get_candidate_words():
            Returns a list of candidate words sorted alphabetically.

        get_candidate_connections():
            Returns a list of candidate connections with the part of speech tag stripped.

        __repr__():
            Returns a string representation of the ConnectionGroup instance.

        __eq__(other):
            Determines if the group is equal to another group based on candidate words.
    """

    group_metric: float = field(
        default=0.0,
        metadata={
            "help": "Average cosine similarity of all combinations of words in the group"
        },
    )
    root_word: str = field(default="", metadata={"help": "Root word of the group"})
    candidate_pairs: list = field(
        default_factory=list,
        metadata={"help": "List of candidate word with definition"},
    )
    group_id: Optional[str] = field(
        default=None, metadata={"help": "Checksum identifer for the group"}
    )

    def add_entry(self, word, connection):
        """
        Adds a new entry to the candidate pairs list.

        Args:
            word (str): The word to be added.
            connection (Any): The connection associated with the word.

        Raises:
            ValueError: If the group already contains 4 entries.

        Notes:
            If the addition of the new entry results in the candidate pairs list
            reaching a length of 4, the group ID is computed using the candidate words.
        """
        if len(self.candidate_pairs) < 4:
            self.candidate_pairs.append((word, connection))
            if len(self.candidate_pairs) == 4:
                self.group_id = compute_group_id(self.get_candidate_words())
        else:
            raise ValueError("Group is full, cannot add more entries")

    def get_candidate_words(self):
        """
        Retrieves candidate words from the candidate pairs.

        This method sorts the candidate pairs based on the first element of each pair
        and returns a list of the first elements from the sorted pairs.

        Returns:
            list: A list of candidate words sorted by the first element of each pair.
        """
        sorted_pairs = sorted(self.candidate_pairs, key=lambda x: x[0])
        return [x[0] for x in sorted_pairs]

    def get_candidate_connections(self):
        """
        Retrieve and process candidate connections.

        This method sorts the candidate pairs based on the first element of each pair.
        It then processes the second element of each pair by stripping the part of speech
        tag at the beginning of the connection (e.g., "noun:", "verb:"). If a colon is
        present, it takes the substring after the first colon and strips any leading or
        trailing whitespace. If no colon is present, it leaves the connection as is.

        Returns:
            list: A list of processed candidate connections with part of speech tags removed.
        """
        sorted_pairs = sorted(self.candidate_pairs, key=lambda x: x[0])

        # strip the part of speech tag at the beginning of the connection, which looks like "noun:" or "verb:" etc.
        # find the first colon and take the substring after it
        stripped_connections = [
            x[1].split(":", 1)[1].strip() if ":" in x[1] else x[1] for x in sorted_pairs
        ]

        return stripped_connections

    def __repr__(self):
        """
        Returns a string representation of the object, including the group metric,
        root word, group ID, candidate group words, and candidate connections.

        Returns:
            str: A formatted string representation of the object.
        """
        return_string = f"group metric: {self.group_metric}, "
        return_string += f"root word: {self.root_word}, group id: {self.group_id}\n"
        return_string += f"candidate group: {self.get_candidate_words()}\n"
        for connection in self.get_candidate_connections():
            return_string += f"\t{connection}\n"

        return return_string

    # method to determine if the group is equal to another group
    def __eq__(self, other):
        """
        Compare two objects for equality based on their candidate words.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the sets of candidate words from both objects are equal, False otherwise.
        """
        return set(self.get_candidate_words()) == set(other.get_candidate_words())


@dataclass
class RecommendedGroup:
    """
    A class to represent a recommended group of words with a connection description.

    Attributes:
    ----------
    words : List[str]
        A list of words that form the recommended group.
    connection_description : str
        A description of the connection between the words in the recommended group.

    Methods:
    -------
    __repr__():
        Returns a string representation of the RecommendedGroup instance.
    """

    words: List[str]
    connection_description: str

    def __repr__(self):
        """
        Returns a string representation of the object, including the recommended group of words
        and the connection description.

        Returns:
            str: A formatted string containing the recommended group of words and the connection description.
        """
        return f"Recommended Group: {self.words}\nConnection Description: {self.connection_description}"


@weave.op()
async def setup_puzzle(state: PuzzleState) -> PuzzleState:
    """
    Asynchronously sets up the puzzle state by initializing various parameters, generating vocabulary and embeddings,
    and storing them in a SQLite database.

    Args:
        state (PuzzleState): The current state of the puzzle.

    Returns:
        PuzzleState: The updated state of the puzzle.

    Side Effects:
        - Logs entry and exit points of the function.
        - Prints status messages to the console.
        - Initializes various state parameters.
        - Generates vocabulary and embeddings for the remaining words.
        - Stores the vocabulary and embeddings in a SQLite database.

    Raises:
        Any exceptions raised by the asynchronous operations or database interactions.
    """
    logger.info("Entering setup_puzzle:")
    logger.debug(f"\nEntering setup_puzzle State: {pp.pformat(state)}")

    state["current_tool"] = "setup_puzzle"
    print(f"\nENTERED {state['current_tool'].upper()}")

    # initialize the state
    state["puzzle_status"] = "initialized"
    state["tool_status"] = "initialized"
    state["invalid_connections"] = []
    state["mistake_count"] = 0
    state["found_count"] = 0
    state["recommendation_count"] = 0
    state["llm_retry_count"] = 0
    state["recommended_words"] = []

    print(f"Puzzle Words: {state['words_remaining']}")

    # generate vocabulary for the words
    print("\nGenerating vocabulary for the words...this may take about a minute")
    vocabulary = await generate_vocabulary(state["words_remaining"])

    # Convert dictionary to DataFrame
    rows = []
    for word, definitions in vocabulary.items():
        for definition in definitions:
            rows.append({"word": word, "definition": definition})
    df = pd.DataFrame(rows)

    # Generate embeddings
    print("\nGenerating embeddings for the definitions")
    embeddings = generate_embeddings(df["definition"].tolist())
    # convert embeddings to json strings for storage
    df["embedding"] = [json.dumps(v) for v in embeddings]

    async with aiosqlite.connect(state["vocabulary_db_fp"]) as conn:
        async with db_lock:
            cursor = await conn.cursor()
            # create the table
            await cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vocabulary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT,
                    definition TEXT,
                    embedding TEXT
                )
                """
            )
            await conn.executemany(
                "INSERT INTO vocabulary (word, definition, embedding) VALUES (?, ?, ?)",
                df.values.tolist(),
            )
            await conn.commit()
            # await conn.close()

    logger.info("Exiting setup_puzzle:")
    logger.debug(f"\nExiting setup_puzzle State: {pp.pformat(state)}")

    return state


SYSTEM_MESSAGE_VOCAB = SystemMessage(
    """
You are an expert in language and knowledgeable on how words are used.

Your task is to generate as many diverse definitions as possible for the given word.  Follow these steps:

1. come up with a list of all possible parts of speech that the given word can be,e.g., noun, verb, adjective, etc.
2. for each part of speech, generate one or more examples of the given word for that parts of speech.  preappend the part of speech to the examples, e.g., "noun: example1", "verb: example2", etc.
3. combine all examples into a single list.

Return your response as a JSON object with the word as the key and the connotations as a list of strings.

example:

{
  "word": [
    "noun: example1", 
    "noun: example2", 
    "adjective: example3",]
}
"""
)


@weave.op()
async def generate_vocabulary(words, model="gpt-4o", temperature=0.7, max_tokens=4096):
    """
    Asynchronously generates a vocabulary dictionary for a list of words using the GPT-4o model.

    Args:
        words (list of str): A list of words for which to generate vocabulary entries.
        model (str, optional): The model to use for generating vocabulary. Defaults to "gpt-4o".
        temperature (float, optional): The temperature to use for the model. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 4096.

    Returns:
        dict: A dictionary where keys are the input words and values are the generated vocabulary entries.

    """
    # Initialize the OpenAI LLM with your API key and specify the GPT-4o model
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    vocabulary = {}

    async def process_word(the_word):
        prompt = f"\n\ngiven word: {the_word}"
        prompt = HumanMessage(prompt)
        prompt = [SYSTEM_MESSAGE_VOCAB, prompt]
        result = await llm.ainvoke(prompt)
        vocabulary[the_word] = json.loads(result.content)[the_word]

    await asyncio.gather(*[process_word(word) for word in words])

    return vocabulary


@weave.op()
def generate_embeddings(definitions, model="text-embedding-3-small"):
    """
    Generate embeddings for a list of definitions using a specified embedding model.

    Args:
        definitions (list of str): A list of text definitions to be embedded.
        model (str, optional): The name of the embedding model to use. Defaults to "text-embedding-3-small".

    Returns:
        list of list of float: A list of embeddings, where each embedding is a list of floats.
    """

    # setup embedding model
    embed_model = OpenAIEmbeddings(model=model)

    embeddings = embed_model.embed_documents(definitions)

    return embeddings


PLANNER_SYSTEM_MESSAGE = """
    You are an expert in managing the sequence of a workflow. Your task is to
    determine the next tool to use given the current state of the workflow.

    the eligible tools to use are: ["setup_puzzle", "get_llm_recommendation", "apply_recommendation", "get_embedvec_recommendation", "END"]

    The important information for the workflow state is to consider are: "puzzle_status", "tool_status", and "current_tool".

    Using the provided instructions, you will need to determine the next tool to use.

    output response in json format with key word "tool" and the value as the output string.
    
"""


@weave.op()
def ask_llm_for_next_step(
    instructions, puzzle_state, model="gpt-3.5-turbo", temperature=0, max_tokens=4096
):
    """
    Asks the language model (LLM) for the next step based on the provided prompt.

    Args:
        prompt (AIMessage): The prompt containing the content to be sent to the LLM.
        model (str, optional): The model to be used by the LLM. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): The temperature setting for the LLM, controlling the randomness of the output. Defaults to 0.
        max_tokens (int, optional): The maximum number of tokens for the LLM response. Defaults to 4096.

    Returns:
        AIMessage: The response from the LLM containing the next step.
    """
    logger.info("Entering ask_llm_for_next_step")
    logger.debug(f"Entering ask_llm_for_next_step Instructions: {instructions.content}")
    logger.debug(f"Entering ask_llm_for_next_step Prompt: {puzzle_state.content}")

    # Create a prompt by concatenating the system and human messages
    conversation = [PLANNER_SYSTEM_MESSAGE, instructions, puzzle_state]

    logger.debug(f"conversation: {pp.pformat(conversation)}")

    # Invoke the LLM
    response = chat_with_llm(
        conversation,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.debug(f"response: {pp.pformat(response)}")

    logger.info("Exiting ask_llm_for_next_step")
    logger.info(f"exiting ask_llm_for_next_step response {response.content}")

    return response


VALIDATOR_SYSTEM_MESSAGE = SystemMessage(
    """
    anaylyze the following set of "candidate group" of 4 words.
    
    For each  "candidate group"  determine if the 4 words are connected by a single theme or concept.

    eliminate "candidate group" where the 4 words are not connected by a single theme or concept.

    return the "candidate group" that is unlike the other word groups

    if there is no  "candidate group" connected by a single theme or concept, return the group with the highest group metric.

    return response in json with the
    * key "candidate_group" for the "candidate group" that is connected by a single theme or concept that is the most unique about the "candidate group".  This is a list of 4 words.
    * key "explanation" with a few word summary for the reason for the response.
    """
)


def choose_embedvec_item(candidates, model="gpt-4o", temperature=0.7, max_tokens=4096):
    """
    Selects a response from a list of candidate messages generated by embedvec tool using a specified language model.

    Args:
        candidates (str): The input text containing candidate messages to be evaluated.
        model (str, optional): The model name to be used for the language model. Defaults to "gpt-4o".
        temperature (float, optional): The sampling temperature to use. Higher values mean the model will take more risks. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 4096.

    Returns:
        dict: The selected response in JSON format.
    """

    # Initialize the OpenAI LLM with your API key and specify the GPT-4o model
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    prompt = HumanMessage(candidates)
    prompt = [VALIDATOR_SYSTEM_MESSAGE, prompt]
    result = llm.invoke(prompt)

    return json.loads(result.content)


@weave.op()
def get_candidate_words(df: pd.DataFrame) -> list:
    """
    Generate a list of candidate word groups based on cosine similarity of their embeddings.

    Args:
        df (pd.DataFrame): DataFrame containing words and their corresponding embeddings. Dataframe should have two columns: 'word', 'definition' and 'embedding', in that order.

    Returns:
        list: A list of unique candidate word groups sorted by their group metric in descending order.
    """

    candidate_list = []

    # create cosine similarity matrix for all pairs of the vectors
    cosine_similarities = cosine_similarity(df["embedding"].tolist())
    print(cosine_similarities.shape)

    # for each row in the cosine similarity matrix, sort by the cosine similarity
    sorted_cosine_similarites = np.argsort(cosine_similarities, axis=1)
    print(sorted_cosine_similarites.shape)

    # group of words that are most similar to each other
    for r in range(df.shape[0]):

        # get the top 3 closest words that are not the same as the current word and are not already connected
        connected_words = set()
        top3 = []
        for i in range(sorted_cosine_similarites.shape[1] - 2, 0, -1):
            c = sorted_cosine_similarites[r, i]

            # make sure the word is not already connected and not the current word
            if df.iloc[c, 0] not in connected_words and df.iloc[c, 0] != df.iloc[r, 0]:
                connected_words.add(df.iloc[c, 0])
                top3.append(c)
            if len(connected_words) == 3:
                break

        # create candidate group for the current word and the top 3 closest words
        if df.iloc[r, 0] not in connected_words and len(connected_words) == 3:
            candidate_group = ConnectionGroup()
            candidate_group.group_metric = cosine_similarities[r, top3].mean()
            candidate_group.root_word = df.iloc[r, 0]
            candidate_group.add_entry(df.iloc[r, 0], df.iloc[r, 1])

            for c in top3:
                candidate_group.add_entry(df.iloc[c, 0], df.iloc[c, 1])

            combinations = list(itertools.combinations([r] + top3, 2))
            candidate_group.group_metric = np.array(
                [cosine_similarities[r, c] for r, c in combinations]
            ).mean()

            candidate_list.append(candidate_group)

    # sort the candidate list by the group metric in descending order
    candidate_list.sort(key=lambda x: x.group_metric, reverse=True)

    # remove duplicate groups
    found_groups = set()
    unique_candidate_list = []
    for candidate in candidate_list:
        if candidate.group_id not in found_groups:
            unique_candidate_list.append(candidate)
            found_groups.add(candidate.group_id)

    return unique_candidate_list


ANCHOR_WORDS_SYSTEM_PROMPT = (
    "you are an expert in the nuance of the english language.\n\n"
    "You will be given three words. you must determine if the three words can be related to a single topic.\n\n"
    "To make that determination, do the following:\n"
    "* Determine common contexts for each word. \n"
    "* Determine if there is a context that is shared by all three words.\n"
    "* respond 'single' if a single topic can be found that applies to all three words, otherwise 'multiple'.\n"
    "* Provide an explanation for the response.\n\n"
    "return response in json with the key 'response' with the value 'single' or 'multiple' and the key 'explanation' with the reason for the response."
)

CREATE_GROUP_SYSTEM_PROMPT = """
you will be given a list called the "anchor_words".

You will be given list of "candidate_words", select the one word that is most higly connected to the "anchor_words".

Steps:
1. First identify the common connection that is present in all the "anchor_words".  If each word has multiple meanings, consider the meaning that is most common among the "anchor_words".

2. Now test each word from the "candidate_words" and decide which one has the highest degree of connection to the "anchor_words".    

3. Return the word that is most connected to the "anchor_words" and the reason for its selection in json structure.  The word should have the key "word" and the explanation should have the key "explanation".
"""


@weave.op()
def one_away_analyzer(
    state: PuzzleState, one_away_group: List[str], words_remaining: List[str]
) -> List[Tuple[str, List[str]]]:
    """
    Analyzes a group of words that are one step away from being a valid group and
    attempts to find a single-topic group among them. If found, it recommends a
    new group by adding one more word to the selected single-topic group.

    Args:
        state (PuzzleState): The current state of the puzzle, including found and mistake counts.
        one_away_group (List[str]): A list of words that are one step away from being a valid group.
        words_remaining (List[str]): A list of words remaining to be tested.

    Returns:
        List[Tuple[str, List[str]]]: A recommended group of words that form a single topic,
        along with a description of the connection, or None if no such group is found.
    """
    print("\nENTERED ONE-AWAY ANALYZER")
    print(
        f"found count: {state['found_count']}, mistake_count: {state['mistake_count']}"
    )

    single_topic_groups = []
    possible_anchor_words_list = list(itertools.combinations(one_away_group, 3))

    for anchor_list in possible_anchor_words_list:
        # determine if the anchor words can be related to a single topic
        anchor_words = "\n\n" + ", ".join(anchor_list)
        prompt = [SystemMessage(ANCHOR_WORDS_SYSTEM_PROMPT), HumanMessage(anchor_words)]
        response = chat_with_llm(prompt)
        response = json.loads(response.content)

        logger.info(f"\n>>>Anchor Words: {anchor_list}")
        logger.info(response)

        if response["response"] == "single":

            single_topic_groups.append(
                RecommendedGroup(
                    words=anchor_list, connection_description=response["explanation"]
                )
            )

    print(f"\n>>>Number of single topic groups: {len(single_topic_groups)}")
    if len(single_topic_groups) > 1:
        # if more than one single topic group is found, select one at random
        print(
            f"More than one single-topic group recommendations, selecting one at random."
        )
        selected_word_group = random.choice(single_topic_groups)
    elif len(single_topic_groups) == 1:
        # if only one single topic group is found, select that one
        print(f"Only one single-topic group recommendation found.")
        selected_word_group = single_topic_groups[0]
    else:
        # if no single topic groups are found, select None
        print(f"No single-topic group recommendations found.")
        selected_word_group = None

    if selected_word_group:
        print(f"\n>>>Selected single-topic group:\n{selected_word_group}")
        # remove original one-away invalid group from the remaining word list
        words_to_test = [x for x in words_remaining if x not in one_away_group]
        user_prompt = "\n\nanchor_words: " + ", ".join(selected_word_group.words)
        user_prompt += "\n\n" + "candidate_words: " + ", ".join(words_to_test)
        logger.info(f"single-topic user prompt:\n {user_prompt}")

        prompt = [SystemMessage(CREATE_GROUP_SYSTEM_PROMPT), HumanMessage(user_prompt)]

        response = chat_with_llm(prompt)
        response = json.loads(response.content)
        logger.info(response)
        new_group = list(selected_word_group.words) + [response["word"]]
        one_away_group_recommendation = RecommendedGroup(
            words=new_group, connection_description=response["explanation"]
        )
        print(f"\n>>>One-away group recommendations:")
        logger.info(one_away_group_recommendation)
    else:
        # if no single topic groups are found, single None
        one_away_group_recommendation = None

    return one_away_group_recommendation


@weave.op()
async def run_agentic_solver(words, solution):
    """
    Runs the agentic solver to solve a puzzle using a state graph workflow.

    Args:
        words (list): A list of words to be used in the puzzle.
        solution (dict): A dictionary containing the solution groups and their reasons.

    Returns:
        list: A list of correct recommendation groups found during the workflow execution.

    The function sets up a state graph workflow with various nodes and edges to solve the puzzle.
    It uses a temporary database to store the puzzle state and runs the workflow in an asynchronous
    manner. The workflow involves setting up the puzzle, getting recommendations, applying them,
    and checking the solutions. The function returns all the correct recommendation groups found.
    """
    # result = workflow_graph.invoke(initial_state, runtime_config)

    workflow = StateGraph(PuzzleState)

    workflow.add_node("run_planner", run_planner)
    workflow.add_node("setup_puzzle", setup_puzzle)
    workflow.add_node("get_embedvec_recommendation", get_embedvec_recommendation)
    workflow.add_node("get_llm_recommendation", get_llm_recommendation)
    workflow.add_node("apply_recommendation", apply_recommendation)

    workflow.add_conditional_edges(
        "run_planner",
        determine_next_action,
        {
            "setup_puzzle": "setup_puzzle",
            "get_embedvec_recommendation": "get_embedvec_recommendation",
            "get_llm_recommendation": "get_llm_recommendation",
            "apply_recommendation": "apply_recommendation",
            END: END,
        },
    )

    workflow.add_edge("setup_puzzle", "run_planner")
    workflow.add_edge("get_llm_recommendation", "run_planner")
    workflow.add_edge("get_embedvec_recommendation", "run_planner")
    workflow.add_edge("apply_recommendation", "run_planner")

    workflow.set_entry_point("run_planner")

    memory_checkpoint = MemorySaver()

    workflow_graph = workflow.compile(
        checkpointer=memory_checkpoint,
        interrupt_before=["setup_puzzle", "apply_recommendation"],
    )
    # workflow_graph.get_graph().draw_png("images/connection_solver_embedvec_graph.png")

    runtime_config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 50,
    }

    def check_one_solution(solution, gen_words, gen_reason):
        for sol_dict in solution["groups"]:
            sol_words = sol_dict["words"]
            sol_reason = sol_dict["reason"]
            if set(gen_words) == set(sol_words):
                print(f"{gen_reason} ~ {sol_reason}: {gen_words} == {sol_words}")
                return "correct"
            elif len(set(gen_words).intersection(set(sol_words))) == 3:
                return "one_away"
        else:
            return "incorrect"

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        print(f"Using temporary database: {tmp_db.name}")

        initial_state = PuzzleState(
            puzzle_status="",
            current_tool="",
            tool_status="",
            workflow_instructions=None,
            llm_temperature=0.7,
            vocabulary_db_fp=tmp_db.name,
            recommendation_correct_groups=[],
        )

        # run part of workflow to do puzzle setup
        async for chunk in workflow_graph.astream(
            initial_state, runtime_config, stream_mode="values"
        ):
            pass

        # continue workflow until feedback is needed on a group recommendation
        while chunk["tool_status"] != "puzzle_completed":
            current_state = workflow_graph.get_state(runtime_config)
            logger.debug(f"\nCurrent state: {current_state}")
            logger.info(f"\nNext action: {current_state.next}")

            if current_state.next[0] == "setup_puzzle":
                # setup puzzle for the workflow
                workflow_graph.update_state(
                    runtime_config,
                    {
                        "words_remaining": words,
                    },
                )

            elif current_state.next[0] == "apply_recommendation":
                # check the recommendation against the solutions
                checker_response = check_one_solution(
                    solution,
                    chunk["recommended_words"],
                    chunk["recommended_connection"],
                )

                workflow_graph.update_state(
                    runtime_config,
                    {
                        "puzzle_checker_response": checker_response,
                    },
                )
            else:
                raise RuntimeError(f"Unexpected next action: {current_state.next[0]}")

            # run rest of workflow until feedback is needed on a group recommendation
            async for chunk in workflow_graph.astream(
                None, runtime_config, stream_mode="values"
            ):
                logger.debug(f"\nstate: {workflow_graph.get_state(runtime_config)}")
                pass

    # return all found correct groups
    return chunk["recommendation_correct_groups"]


# used for testing and debugging
async def run_agentic_simulator(words, solution):
    answer = [g for g in solution["groups"]]
    return answer


# For testing run the module, e.g.,
# python agentic_tools.py
async def main():
    print(f"Running Connection Solver Agent {__version__}")

    parser = argparse.ArgumentParser(
        description="Set logging level for the application."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # argument for numbr of puzzle setups to run
    parser.add_argument(
        "--num_puzzles",
        type=int,
        default=2,
        help="Number of puzzles to run",
    )

    # Parse arguments
    args = parser.parse_args()

    print(f"Solving for {args.num_puzzles} puzzles")

    # Configure logging
    configure_logging(args.log_level)

    # Create a logger instance
    logger = logging.getLogger(__name__)

    def load_jsonl(file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line))
        return data

    # load the data
    puzzle_setups = load_jsonl("connections_prompts.jsonl")

    print(f"Number of prompts: {len(puzzle_setups)}")

    found_groups_list = []
    for i, puzzle_setup in enumerate(puzzle_setups[: args.num_puzzles]):
        print(f"\nSolving puzzle {i+1} with words: {puzzle_setup['words']}")

        found_groups = await run_agentic_solver(
            puzzle_setup["words"], puzzle_setup["solution"]
        )
        found_groups_list.append(found_groups)

    print("\nAll Found Groups:")
    for i, found_groups in enumerate(found_groups_list):
        print("")
        for found_group in found_groups:
            print(f"Group {i+1}, {found_group}")


if __name__ == "__main__":
    asyncio.run(main())
