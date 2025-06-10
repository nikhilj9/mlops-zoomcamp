from prefect import flow, task

@task
def say_hello():
    """A simple task that says hello."""
    print("Hello, Prefect!")

@flow
def hello_world_flow():
    """Our main flow that calls the hello task."""
    say_hello()

if __name__ == "__main__":
    hello_world_flow()