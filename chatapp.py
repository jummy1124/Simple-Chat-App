import torch
import tkinter as tk
from tkinter import ttk
from transformers import pipeline


# Create a simple GUI interface
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat App")
        self.root.geometry("400x500")

        # Chat history display area
        self.chat_history = tk.Text(self.root, state='normal', width=50, height=20)
        self.chat_history.pack(pady=10)

        # Model selection area
        self.model_label = tk.Label(self.root, text="Select Model:")
        self.model_label.pack()

        self.model_choice = ttk.Combobox(self.root, values=["Llama-3.2-1B", "Llama-3.2-3B"])
        self.model_choice.set("Llama-3.2-1B")
        self.model_choice.pack()

        # Input box
        self.entry = tk.Entry(self.root, width=50)
        self.entry.pack(pady=10)

        # Send button
        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack()

        # Clear screen button
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_chat)
        self.clear_button.pack(pady=5)

        # Initialize the Hugging Face model
        self.model_choice.set("Llama-3.2-1B")
        # self.generator_bloomz = pipeline("text-generation", model="bigscience/bloomz-1b1")
        self.generator_llama1b = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
        self.generator_llama3b = pipeline("text-generation", model="meta-llama/Llama-3.2-3B")

    def send_message(self):
        user_message = self.entry.get()
        if user_message:
            # Show user information above
            self.entry.delete(0, tk.END)

            selected_model = self.model_choice.get()
            if selected_model == "Llama-3.2-1B":
                response = self.get_huggingface_response(self.generator_llama1b, user_message)
            elif selected_model == "Llama-3.2-3B":
                response = self.get_huggingface_response(self.generator_llama3b, user_message)

            # Show model response above
            self.display_message(response)

    def display_message(self, message):
        self.chat_history.config(state='normal')
        self.chat_history.insert("1.0", message + "\n")  # Insert new message on top
        self.chat_history.config(state='disabled')

        # Control the number of messages displayed and automatically delete the oldest messages when there are too many messages
        self.remove_old_messages()

    def get_huggingface_response(self, generator, user_message):
        # Generate responses using Hugging Face Transformers
        try:
            response = generator(user_message, max_length=100, num_return_sequences=1)
            return response[0]['generated_text'].strip()
        except Exception as e:
            return f"Error: Unable to generate response - {str(e)}"

    def clear_chat(self):
        # Clear chat history
        self.chat_history.config(state='normal')
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state='disabled')

    def remove_old_messages(self):
        # Set the maximum number of display lines
        max_lines = 20
        content = self.chat_history.get("1.0", tk.END)
        lines = content.splitlines()

        if len(lines) > max_lines:
            # Delete more than one old message
            self.chat_history.delete(f"{len(lines) - max_lines + 1}.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
