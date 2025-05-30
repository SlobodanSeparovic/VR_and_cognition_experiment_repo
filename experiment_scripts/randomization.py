import random
import csv

def randomize_experiment_order(participants):
    """Generate randomized order for each participant"""
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
        
        # Shuffle versions independently for 2-back and 3-back tasks
        available_2back_versions = random.sample(versions, len(versions))  # Shuffle versions for 2-back
        available_3back_versions = random.sample(versions, len(versions))  # Shuffle versions for 3-back
        
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

def save_random_seq_csv(randomized_orders, filename):
    """Save randomized order to a CSV file"""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Header row
        writer.writerow([
            'participant_id',
            'baseline_order',
            'environment',
            'task',
            'version'
        ])
        
        # Data rows
        for participant, data in randomized_orders.items():
            baseline_str = ' > '.join(data['baseline'])
            writer.writerow([
                participant,
                baseline_str
                ])
            
            for(environment,task), version in data['task_assignments'].items():
                writer.writerow([
                    '',
                    '',
                    environment,
                    task,
                    version
                    ])

if __name__ == "__main__":
    # Create 60 participants
    participants = [f'Participant {i+1}' for i in range(60)]
    randomized_orders = randomize_experiment_order(participants)

    # Save to CSV 
    save_random_seq_csv(randomized_orders, 'random_seqs.csv')

    print("Randomized sequences saved to 'random_seqs.csv'.")