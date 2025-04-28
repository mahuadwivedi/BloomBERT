import json

# Function to collect user feedback after every chatbot response
def collect_feedback():
    feedback_list = []
    frequency_list = []
    
    while True:
        # Get user feedback input
        feedback = input("Enter feedback (+1 for positive, -1 for negative, 0 for neutral or 'quit' to stop): ")
        
        # If user wants to stop inputting, break the loop
        if feedback.lower() == 'quit':
            break
        
        # Ensure valid input (+1, -1, 0)
        if feedback not in ['1', '-1', '0']:
            print("Invalid input! Please enter +1, -1, or 0.")
            continue
        
        # Convert feedback to integer
        feedback = int(feedback)
        
        # Add feedback and frequency (which is always 1 for each individual response)
        feedback_list.append(feedback)
        frequency_list.append(1)  # Each feedback is from a single user (or single instance)
    
    return feedback_list, frequency_list

# Function to calculate the weighted average reward
def calculate_weighted_avg(feedback, frequencies):
    weighted_sum = 0
    total_frequency = 0
    
    # Calculate the weighted sum and total frequency
    for i in range(len(feedback)):
        weighted_sum += feedback[i] * frequencies[i]
        total_frequency += frequencies[i]
    
    # Calculate the weighted average reward
    weighted_avg_reward = weighted_sum / total_frequency if total_frequency != 0 else 0
    return weighted_avg_reward

# Function to save feedback data to a file
def save_feedback_to_file(feedback_list, frequency_list, filename="feedback_data.json"):
    feedback_data = {"feedback": feedback_list, "frequency": frequency_list}
    with open(filename, 'w') as f:
        json.dump(feedback_data, f)

# Function to load feedback data from a file
def load_feedback_from_file(filename="feedback_data.json"):
    try:
        with open(filename, 'r') as f:
            feedback_data = json.load(f)
            return feedback_data['feedback'], feedback_data['frequency']
    except FileNotFoundError:
        return [], []  # Return empty lists if file doesn't exist

# Main function to execute the RLHF process
def main():
    print("Welcome to the RLHF Feedback System!")
    
    # Load existing feedback data if available
    feedback_list, frequency_list = load_feedback_from_file()
    
    # Collect new feedback from the user
    new_feedback, new_frequencies = collect_feedback()
    
    # Append new feedback to the existing data
    feedback_list.extend(new_feedback)
    frequency_list.extend(new_frequencies)
    
    # Calculate the weighted average reward
    avg_reward = calculate_weighted_avg(feedback_list, frequency_list)
    print(f"Weighted Average Reward: {avg_reward}")
    
    # Save the updated feedback data
    save_feedback_to_file(feedback_list, frequency_list)
    
    print("Feedback saved successfully!")

if __name__ == "__main__":
    main()
