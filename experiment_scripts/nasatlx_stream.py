import tkinter as tk
from pylsl import StreamInfo, StreamOutlet
import time

# Define LSL stream (with 6 channels)
info = StreamInfo('NASA_TLX_Questionnaire', 'Responses', 6, 1, 'int32', 'survey123')

# Add channels
info.desc().append_child("channels")  # Create channels node
channels = [
    {'label': 'MENTAL_DEMAND', 'unit': 'integer'},
    {'label': 'PHYSICAL_DEMAND', 'unit': 'integer'},
    {'label': 'TEMPORAL_DEMAND', 'unit': 'integer'},
    {'label': 'PERFORMANCE', 'unit': 'integer'},
    {'label': 'EFFORT', 'unit': 'integer'},
    {'label': 'FRUSTRATION', 'unit': 'integer'}
]

# Add channels to LSL info
channels_node = info.desc().child('channels')
for channel in channels:
    channel_element = channels_node.append_child('channel')  # Create new channel element
    channel_element.append_child_value('label', channel['label'])  # Add the label
    channel_element.append_child_value('unit', channel['unit'])  # Add the unit

# Create an outlet for the stream
outlet = StreamOutlet(info)

# Define questions
questions = [
    "MENTAL DEMAND:\nHow mentally demanding was the task?",
    "PHYSICAL DEMAND:\nHow physically demanding was the task?",
    "TEMPORAL DEMAND:\nHow hurried or rushed was the pace of the task?",
    "PERFORMANCE:\nHow successful were you in accomplishing what you were asked to do?",
    "EFFORT:\nHow hard did you have to work to accomplish your level of performance?",
    "FRUSTRATION:\nHow insecure, discouraged, irritated, stressed, and annoyed were you?"
]

# Labels for radio buttons
labels = ["Very Low"] + [" " for _ in range(19)] + ["Very High"]
performance_labels = ["Perfect   "] + [" " for _ in range(19)] + ["   Failure"]

# Tkinter UI
root = tk.Tk()
root.title("NASA TLX Questionnaire")
root.geometry("600x700")

# Adjust the responses to work with a 0-100 scale with increments of 5
responses = [tk.IntVar(value=50) for _ in questions]  # Default to midpoint (50)

def create_radio_buttons(question, index):
    frame = tk.Frame(root)
    frame.pack(pady=12, anchor='center', fill='x')
    
    # Display the channel name in bold and the question text centered with spacing before the response optionsgit push
    label_lines = question.strip().split('\n', 1)
    tk.Label(frame, text=label_lines[0], font=("Arial", 28, "bold")).pack()
    tk.Label(frame, text=label_lines[1], font=("Arial", 27), wraplength=700, justify="center").pack()
    # tk.Label(frame, text="").pack()

    
    button_frame = tk.Frame(frame)
    button_frame.pack()

    # Add left label on the far left
    left_label = performance_labels[0] if index == 3 else labels[0]
    tk.Label(button_frame, text=left_label, font=("Arial", 27)).pack(side="left", padx=20)

    # Add the radio buttons in the middle with values 0, 5, 10, ..., 100
    for i in range(21):
        value = i * 5  # This gives values from 0 to 100 in increments of 5
        label = "I"
        font_size = 23
        
        if value == 50:  # Middle value
            font_size = 29  # Larger font for the middle value
        
        rb = tk.Radiobutton(button_frame, variable=responses[index], value=value, indicatoron=False, width=3, text=label, font=("Arial", font_size))
        rb.pack(side="left", padx=2)

    # Add right label on the far right
    right_label = performance_labels[-1] if index == 3 else labels[-1]
    tk.Label(button_frame, text=right_label, font=("Arial", 27)).pack(side="right", padx=20)

# Generate UI
for i, question in enumerate(questions):
    create_radio_buttons(question, i)

# Submit function
def submit_responses():
    sample = [var.get() for var in responses]
    print(f"Sending: {sample}")
    outlet.push_sample(sample)
    
    for widget in root.winfo_children():
        widget.pack_forget()
    
    tk.Label(root, text="Thank you for completing the questionnaire!\nPress escape button to exit.", font=("Arial", 29)).pack(pady=20)

# Submit button
tk.Button(root, text="Submit", font=("Arial", 29), command=submit_responses).pack(pady=20)

# Function to close the window (bound to the Escape key)
def close_window(event):
    root.quit()

# Bind the Escape key to close the window
root.bind('<Escape>', close_window)

root.mainloop()