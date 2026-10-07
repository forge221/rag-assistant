import os
import sys

# A packaged windowed app has no terminal. Some libraries print messages anyway,
# which crashes if there's nowhere to print. Send that output to nowhere instead.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


import queue
import threading
from tkinter import filedialog, messagebox

import customtkinter as ctk

from rag_engine import RagEngine


class RagApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("RAG Assistant")
        self.geometry("900x650")
        self.minsize(600, 400)

        self.events = queue.Queue()
        self.busy = False
        self.last_question = ""

        self.engine = RagEngine(status=lambda message: self.events.put(("status", message)))

        self._build_layout()
        self._set_busy(True)
        self._run_in_background(self._startup)
        self.after(100, self._process_events)

    # ---- Layout ----

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        self.add_button = ctk.CTkButton(toolbar, text="Add documents", command=self._on_add_documents)
        self.add_button.pack(side="left", padx=5, pady=5)

        self.reset_button = ctk.CTkButton(toolbar, text="New conversation", command=self._on_reset)
        self.reset_button.pack(side="left", padx=5, pady=5)

        self.docs_label = ctk.CTkLabel(toolbar, text="Documents: -")
        self.docs_label.pack(side="left", padx=15)

        self.chat = ctk.CTkTextbox(self, wrap="word")
        self.chat.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.chat.configure(state="disabled")

        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(input_frame, placeholder_text="Ask a question about your documents...")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.entry.bind("<Return>", lambda event: self._on_send())

        self.send_button = ctk.CTkButton(input_frame, text="Send", width=80, command=self._on_send)
        self.send_button.grid(row=0, column=1)

        self.status_label = ctk.CTkLabel(self, text="Starting...", anchor="w")
        self.status_label.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 8))

    # ---- Helpers ----

    def _append(self, text: str):
        self.chat.configure(state="normal")
        self.chat.insert("end", text + "\n\n")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _set_busy(self, busy: bool):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (self.entry, self.send_button, self.add_button, self.reset_button):
            widget.configure(state=state)
        if not busy:
            self.entry.focus()

    def _update_docs_label(self):
        count = len(self.engine.document_names())
        self.docs_label.configure(text=f"Documents: {count}")

    def _run_in_background(self, task, *args):
        def worker():
            try:
                task(*args)
            except Exception as error:
                self.events.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    # ---- Background tasks (run on a separate thread, never touch the window) ----

    def _startup(self):
        ok, message = self.engine.check_ollama()
        if not ok:
            self.events.put(("ollama_missing", message))
            return
        self.engine.load()
        self.events.put(("ready", None))

    def _ask(self, question: str):
        result = self.engine.ask(question)
        self.events.put(("answer", result))

    def _add(self, paths):
        added = self.engine.add_documents(paths)
        self.events.put(("docs_added", added))

    # ---- Event handling (runs on the main thread, safe to update the window) ----

    def _process_events(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "status":
                self.status_label.configure(text=payload)

            elif kind == "ready":
                self._update_docs_label()
                if self.engine.document_names():
                    self._append("Ready! Ask a question about your documents.")
                else:
                    self._append("Welcome! Click 'Add documents' to add PDF or text files, then ask a question.")
                self.status_label.configure(text="Ready.")
                self._set_busy(False)

            elif kind == "answer":
                self._show_answer(payload)
                self.status_label.configure(text="Ready.")
                self._set_busy(False)

            elif kind == "docs_added":
                if payload:
                    self._append("Added: " + ", ".join(payload))
                else:
                    self._append("No supported files were selected. Use PDF or .txt files.")
                self._update_docs_label()
                self._set_busy(False)

            elif kind == "ollama_missing":
                self.status_label.configure(text="Ollama not available.")
                self._append(payload + "\n\nAfter fixing this, close and reopen the app.")
                messagebox.showerror("Ollama not found", payload)

            elif kind == "error":
                self._append(f"Something went wrong: {payload}")
                self.status_label.configure(text="Error.")
                self._set_busy(False)

        self.after(100, self._process_events)

    def _show_answer(self, result: dict):
        if result["search_query"] != self.last_question:
            self._append(f"(Searched for: {result['search_query']})")

        self._append(f"Assistant: {result['answer']}")

        if result["sources"]:
            unique_sources = list(dict.fromkeys(source["source"] for source in result["sources"]))
            self._append("Sources: " + ", ".join(unique_sources))

    # ---- Button and key handlers ----

    def _on_send(self):
        if self.busy:
            return
        question = self.entry.get().strip()
        if not question:
            return

        self.last_question = question
        self.entry.delete(0, "end")
        self._append(f"You: {question}")
        self.status_label.configure(text="Thinking...")
        self._set_busy(True)
        self._run_in_background(self._ask, question)

    def _on_add_documents(self):
        if self.busy:
            return
        paths = filedialog.askopenfilenames(
            title="Choose documents",
            filetypes=[("Documents", "*.pdf *.txt"), ("PDF files", "*.pdf"), ("Text files", "*.txt")],
        )
        if not paths:
            return
        self._set_busy(True)
        self._run_in_background(self._add, paths)

    def _on_reset(self):
        if self.busy:
            return
        self.engine.reset_conversation()
        self._append("--- New conversation ---")


if __name__ == "__main__":
    ctk.set_appearance_mode("system")
    app = RagApp()
    app.mainloop()