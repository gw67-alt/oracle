import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import math
import serial
import serial.tools.list_ports # Import this to help find ports
import time

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import math
import serial
import serial.tools.list_ports
import time



# --- Global Configuration ---
ser = None
WAIT_TIMEOUT = 10  # seconds


# Constants
STARTING_CREDITS = 1000
COST_PER_GUESS = 10
WIN_CREDITS = 150
WIN_RANGE = 25 # The +/- range for a winning guess


# --- Serial Communication Function ---
def send_data_to_serial(data_to_send):
    """
    Sends a single character token to the Arduino, waits for a response,
    and reads a line. Returns the received line or None on failure/timeout.
    """
    if not ser or not ser.is_open:
        messagebox.showerror("Serial Error", "Arduino is not connected.")
        return None
    try:
        ser.reset_input_buffer()  # Clear any old data
        ser.write(data_to_send.encode('utf-8'))
        start_time = time.time()
        
        while True:
            # Check for timeout
            if time.time() - start_time > WAIT_TIMEOUT:
                messagebox.showerror("Serial Error", "Timeout waiting for response from Arduino.")
                return None
            
            # Check if there's data in the serial buffer
            if ser.in_waiting > 0:
                received_line = ser.readline().decode('utf-8').strip()
                if received_line:
                    return received_line
            
            # Brief pause to prevent high CPU usage
            time.sleep(0.05)
            
    except serial.SerialException as e:
        messagebox.showerror("Serial Error", f"Communication failed: {e}")
        # Optionally, close the port and disable controls
        ser.close()
        check_button.config(state=tk.DISABLED)
        reset_button.config(state=tk.DISABLED)
        return None

# --- Main Game Application ---
def number_guessing_game():
    global ser # Declare that we are using the global 'ser' variable

    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Guess the Number - Arduino Edition")
    root.geometry("700x550")
    root.resizable(False, False)
    root.configure(bg="#2E2E2E")

    # --- Style Configuration ---
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('.', background='#2E2E2E', foreground='white')
    style.configure('TFrame', background='#2E2E2E')
    style.configure('TLabel', background='#2E2E2E', foreground='white', font=('Arial', 10))
    style.configure('TButton', background='#4A4A4A', foreground='white', font=('Arial', 11, 'bold'), padding=5)
    style.map('TButton', background=[('active', '#6A6A6A'), ('disabled', '#3A3A3A')], foreground=[('disabled', '#777777')])
    style.configure('TEntry', fieldbackground='#4A4A4A', foreground='white', insertcolor='white')
    style.configure('Horizontal.TScale', background='#2E2E2E', troughcolor='#4A4A4A')
    style.configure('Display.TFrame', background='#1A1A1A')
    style.configure('Display.TLabel', background='#1A1A1A', foreground="#FFEB3B", font=('Consolas', 36, 'bold'))

    # --- Game State Dictionary ---
    game_state = {
        "attempts": 0,
        "min_value": 1,
        "max_value": 99,
        "max_attempts": 0,
        "credits": STARTING_CREDITS,
    }

    # --- Game Logic Functions ---
    def check_guess():
        if check_button['state'] == tk.DISABLED:
            return

        try:
            guess = int(round(slider.get()))
            
            # Request a new random number from Arduino for this guess
            received_line = send_data_to_serial('R')
            if not received_line or not received_line.isdigit():
                result_label.config(text="Arduino communication failed.", foreground="#F44336")
                return

            target_number = int(received_line)
            game_state["attempts"] += 1
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}")

            # Check for a win
            if abs(guess - target_number) <= WIN_RANGE:
                game_state["credits"] += WIN_CREDITS
                result_text = f"WIN! The number was {target_number}. You win {WIN_CREDITS} credits!"
                result_label.config(text=result_text, foreground="#4CAF50")
                credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#4CAF50")
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)
                return

            # If not a win, provide feedback and deduct credits
            game_state["credits"] -= COST_PER_GUESS
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FF9800")
            
            # <<< LOGIC FIX >>> - Feedback is based on the actual number, not the range
            if guess < target_number:
                result_label.config(text="Failure! Try again.", foreground="#FFC107")
            else: # guess > target_number
                result_label.config(text="Failure! Try again.", foreground="#FFC107")

            # Check for game over conditions
            if game_state["attempts"] >= game_state["max_attempts"]:
                result_label.config(text=f"No more attempts! The number was {target_number}.", foreground="#F44336")
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)
            elif game_state["credits"] <= 0:
                result_label.config(text="You have no more credits. Game Over!", foreground="#F44336")
                credits_label.config(text="Credits: 0", foreground="#F44336")
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)

        except ValueError:
            result_label.config(text="Invalid input.", foreground="#F44336")

    def setup_game():
        if game_state["credits"] <= 0 and game_state["attempts"] > 0:
            messagebox.showinfo("Game Over", "You are out of credits! Restart the application to play again.")
            return

        try:
            temp_min = int(min_entry.get())
            temp_max = int(max_entry.get())

            if temp_min >= temp_max:
                messagebox.showerror("Invalid Range", "Min must be less than Max.")
                return

            game_state["min_value"] = temp_min
            game_state["max_value"] = temp_max
            range_size = game_state["max_value"] - game_state["min_value"] + 1
            game_state["max_attempts"] = math.ceil(math.log2(range_size)) + 2 # A bit more forgiving

            slider.config(from_=game_state["min_value"], to=game_state["max_value"], state=tk.NORMAL)
            slider.set((game_state["min_value"] + game_state["max_value"]) // 2)
            game_state["attempts"] = 0

            result_label.config(text="", foreground="#FFFFFF")
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}", foreground="#CCCCCC")
            instruction_label.config(text=f"Guess between {game_state['min_value']} and {game_state['max_value']}")
            max_attempts_info_label.config(text=f"({game_state['max_attempts']} attempts max)")
            update_guess_display(slider.get())
            check_button.config(state=tk.NORMAL if ser and ser.is_open else tk.DISABLED)
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B")
        except ValueError:
            messagebox.showerror("Invalid Input", "Min and Max must be valid integers.")

    def update_guess_display(value):
        display_value = int(round(float(value)))
        current_guess_display.config(text=f"{display_value}")

    # --- UI Elements (Widgets) ---
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B")
    credits_label.pack(pady=(10, 5))

    range_frame = ttk.Frame(root, padding="10 5 10 5")
    range_frame.pack(pady=5, fill='x')
    min_label = ttk.Label(range_frame, text="Min:")
    min_label.pack(side=tk.LEFT, padx=(10, 2))
    min_entry = ttk.Entry(range_frame, width=7, font=('Arial', 10), justify='center')
    min_entry.pack(side=tk.LEFT, padx=(0, 10))
    min_entry.insert(0, str(game_state["min_value"]))
    max_label = ttk.Label(range_frame, text="Max:")
    max_label.pack(side=tk.LEFT, padx=(10, 2))
    max_entry = ttk.Entry(range_frame, width=7, font=('Arial', 10), justify='center')
    max_entry.pack(side=tk.LEFT, padx=(0, 10))
    max_entry.insert(0, str(game_state["max_value"]))
    reset_button = ttk.Button(range_frame, text="Set Range / New Game", command=setup_game, width=20)
    reset_button.pack(side=tk.RIGHT, padx=(10, 10))

    instruction_label = ttk.Label(root, text="Set range and click 'Set Range / New Game'", font=('Arial', 11))
    instruction_label.pack(pady=(5, 0))
    max_attempts_info_label = ttk.Label(root, text="", font=('Arial', 9), foreground="#AAAAAA")
    max_attempts_info_label.pack(pady=(0, 10))

    display_frame = ttk.Frame(root, padding="10", relief="sunken", borderwidth=2, style='Display.TFrame')
    display_frame.pack(pady=10, padx=20, fill='x')
    current_guess_display = ttk.Label(display_frame, text="--", style='Display.TLabel', anchor='center')
    current_guess_display.pack(pady=10, fill='x')

    slider = ttk.Scale(root, from_=game_state["min_value"], to=game_state["max_value"], orient=tk.HORIZONTAL, length=400, command=update_guess_display)
    slider.pack(pady=15, padx=30)
    slider.config(state=tk.DISABLED)

    check_button = ttk.Button(root, text="Check Guess", command=check_guess, width=15)
    check_button.pack(pady=5)
    check_button.config(state=tk.DISABLED)

    attempts_label = ttk.Label(root, text="Attempts: 0/0", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(5, 0))
    result_label = ttk.Label(root, text="", font=('Arial', 12, 'bold'), anchor='center', foreground="#FFFFFF", wraplength=450)
    result_label.pack(pady=10, fill='x', padx=20)
    
    # --- Establish Serial Connection AFTER window is ready ---
    port_to_use = "COM4" # Change this if needed
    try:
        # Open the port. This will reset the Arduino.
        ser = serial.Serial(port=port_to_use, baudrate=9600, timeout=5) # 5-second timeout

        # Wait for the Arduino to reset and send its "READY" signal.
        # ser.readline() will wait up to the timeout (5s) for this message.
        startup_message = ser.readline().decode('utf-8').strip()

        if startup_message == "ARDUINO_READY":
            result_label.config(text=f"Successfully connected to Arduino on {port_to_use}", foreground="#4CAF50")
            setup_game() # Automatically set up the first game
        else:
            # This can happen if the wrong program is on the Arduino or there's a timeout
            messagebox.showerror("Connection Error", f"Arduino on {port_to_use} did not send the ready signal. Timed out.")
            result_label.config(text="Arduino handshake failed. Game is disabled.", foreground="#F44336")
            check_button.config(state=tk.DISABLED)

    except serial.SerialException:
        messagebox.showerror("Connection Error", f"Could not open serial port {port_to_use}.\n\n- Is the Arduino plugged in?\n- Is it on the correct COM port?\n- Is the Serial Monitor in the Arduino IDE closed?")
        result_label.config(text="Arduino not found. Game is disabled.", foreground="#F44336")
        check_button.config(state=tk.DISABLED)

    # Start the Tkinter event loop
    root.mainloop()

    # Cleanly close the serial port when the application window is closed
    if ser and ser.is_open:
        ser.close()

# --- Run the game ---
if __name__ == "__main__":
    # The whole application is now encapsulated in this function
    number_guessing_game()
