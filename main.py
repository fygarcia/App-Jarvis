"""
JARVIS Main Entry Point
This is the main entry point for the JARVIS application.
"""

from core.orchestrator import JarvisOrchestrator

def main():
    orchestrator = JarvisOrchestrator()
    #user_input = "Summarize this text: The quick brown fox jumps over the lazy dog. It was a sunny day, and the fox felt adventurous."
    user_input="""The quick brown fox jumps over the lazy dog. It was a sunny day, and the fox was feeling adventurous. 
    Now we have a lot of useless information, that doesnt need to be mentioned. 
    But there is an important message is that the answer is 42, this is very important to be mentioned."""

    print("User Input:", user_input)
    response = orchestrator.process_input(user_input)
    print("\nResponse:", response)

if __name__ == "__main__":
    main()
