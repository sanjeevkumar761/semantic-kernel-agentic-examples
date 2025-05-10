import json


class StateHandler:
    """Handles state lifecycle: loading, updating, and saving."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_state(self) -> dict:
        """Load the state from the JSON file."""
        with open(self.file_path, "r") as file:
            return json.load(file)

    def save_state(self, message: dict) -> None:
        """Save the state to the JSON file."""
        with open(self.file_path, "w") as file:
            json.dump(message, file, indent=4)
        print(f"File '{self.file_path}' updated successfully.")

    def update_state(self, key: str, increment: int) -> dict:
        """Update the value of a key in the message."""
        message = self.load_state()
        message[key] = message.get(key, 0) + increment
        self.save_state(message)
        return message