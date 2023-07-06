import tkinter as tk
from tkinter import filedialog
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from ttkthemes import ThemedTk
from tkinter.ttk import Button, Label, Style
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from api_key import api_key
# Set APIkey for OpenAI Service
os.environ['OPENAI_API_KEY'] = api_key

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        process_pdf(file_path)


def process_pdf(file_path):
    global vectorstore_info, agent_executor, toolkit
    # Create and load PDF Loader
    loader = PyPDFLoader(file_path)
    # Split pages from pdf
    pages = loader.load_and_split()
    # Load documents into vector database aka ChromaDB
    store = Chroma.from_documents(
        pages, embeddings, collection_name='designdesigner')

    # Update vectorstore info object
    vectorstore_info = VectorStoreInfo(
        name="DoCGPT",
        description="Document based GPT",
        vectorstore=store
    )

    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    # Update the label text with the document name
    doc_label.config(text="Based on the: " + os.path.basename(file_path))


def generate_response():
    prompt = prompt_entry.get("1.0", tk.END).strip()

    if prompt:
        response = agent_executor.run(prompt)
        response_text.config(state=tk.NORMAL)
        response_text.delete("1.0", tk.END)
        response_text.insert(tk.END, response)
        response_text.config(state=tk.DISABLED)


# Create themed Tkinter window
window = ThemedTk(theme="equilux")
window.title("DoC GPT")
window.geometry("600x400")

# Customize colors and fonts
background_color = "#121212"
foreground_color = "#00FF00"
entry_background_color = "#000000"
entry_foreground_color = "#FFFFFF"
response_font = ("Courier", 11)

window.configure(bg=background_color)
window.option_add("*TEntry*background", entry_background_color)
window.option_add("*TEntry*foreground", entry_foreground_color)
window.option_add("*TEntry*insertBackground", entry_foreground_color)
window.option_add("*TEntry*font", response_font)
window.option_add("*TLabel*foreground", foreground_color)
window.option_add("*TButton*background", "transparent")
window.option_add("*TButton*foreground", foreground_color)
window.option_add("*TButton*highlightthickness", 0)
window.option_add("*TButton*padding", 0)
window.option_add("*TButton*font", response_font)

# Create style for label boxes
style = Style()
style.configure("Background.TLabel", background=background_color)

# Create GUI components
file_button = Button(window, text="Open PDF", command=open_file)
file_button.pack(pady=10)

doc_label = Label(window, text="Based on the: ", style="Background.TLabel")
doc_label.pack()

prompt_label = Label(window, text="Input your prompt here",
                     style="Background.TLabel")
prompt_label.pack(pady=5)

prompt_entry = tk.Text(
    window, height=5, width=50, bg=entry_background_color, fg=entry_foreground_color, insertbackground=entry_foreground_color)
prompt_entry.pack(pady=5)

generate_button = Button(
    window, text="Generate Response", command=generate_response)
generate_button.pack(pady=10)

response_label = Label(window, text="Response:", style="Background.TLabel")
response_label.pack(pady=5)

response_text = tk.Text(
    window, height=5, width=50, state=tk.DISABLED, bg=entry_background_color, fg=entry_foreground_color, font=response_font)
response_text.pack(pady=5)

# Configure grid weights to make prompt_entry and response_text expand with the window
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(3, weight=1)
window.grid_rowconfigure(6, weight=1)

# Run the Tkinter event loop
window.mainloop()
