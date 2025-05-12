import random

# Function to generate randomized order for each participant
def randomize_experiment_order(participants):
    environments = ['Control', 'Urban', 'Nature']
    tasks = ['2-back', '3-back']
    versions = ['Version 1', 'Version 2', 'Version 3']
    
    randomized_orders = {}
    
    for participant in participants:
        # Randomize order of baseline measurements for the three environments
        baseline_order = random.sample(environments, len(environments))
        
        # Create task-environment pairs (e.g., Control + 2-back, Urban + 3-back, etc.)
        task_environment_pairs = [
            ('Control', '2-back'),
            ('Control', '3-back'),
            ('Urban', '2-back'),
            ('Urban', '3-back'),
            ('Nature', '2-back'),
            ('Nature', '3-back')
        ]
        
        # Randomize the task-environment pairings
        random.shuffle(task_environment_pairs)
        
        # Shuffle versions independently for 1-back and 2-back tasks
        available_2back_versions = random.sample(versions, len(versions))  # Shuffle versions for 1-back
        available_3back_versions = random.sample(versions, len(versions))  # Shuffle versions for 2-back
        
        # Assign versions to the task-environment pairs
        task_assignments = {}
        for (environment, task) in task_environment_pairs:
            if task == '2-back':
                task_assignments[(environment, task)] = available_2back_versions.pop()
            else:
                task_assignments[(environment, task)] = available_3back_versions.pop()

        # Store randomized order for each participant
        randomized_orders[participant] = {
            'baseline': baseline_order,
            'task_assignments': task_assignments
        }
    
    return randomized_orders

# Function to print the randomized order in a readable format
def print_randomized_order(randomized_orders):
    for participant, data in randomized_orders.items():
        print(f"Participant: {participant}")
        print("Baseline Measurement Order:")
        print(f"  {', '.join(data['baseline'])}")
        print("Task Assignments:")
        for (environment, task), version in data['task_assignments'].items():
            print(f"  Environment: {environment}, Task: {task} => {version}")
        print()

if __name__ == "__main__":
    # Ask for the participant number
    participant_number = int(input("Enter the participant number: "))
    
    # Generate a list with a single participant based on the entered number
    participants = [f'Participant {participant_number}']
    
    # Generate randomized order for the participant
    randomized_orders = randomize_experiment_order(participants)
    
    # Print the randomized order for the participant
    print_randomized_order(randomized_orders)