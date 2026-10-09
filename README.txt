RAG Assistant
Ask questions about your own PDF and text documents, answered by AI that
runs entirely on your computer.

=====================================================================
BEFORE FIRST USE (Windows and Mac)
=====================================================================
1. Install Ollama from https://ollama.com
2. Open a terminal (PowerShell on Windows, Terminal on Mac) and run:
      ollama pull llama3.2
3. Make sure Ollama is running (llama icon near the clock on Windows,
   or in the menu bar at the top of the screen on Mac).

=====================================================================
RUNNING ON WINDOWS
=====================================================================
1. Before unzipping, right-click the zip file, choose Properties,
   check "Unblock" at the bottom, and click OK.
   (If there's no Unblock box and Windows blocks the app anyway, open
   PowerShell and run this, using the path to the unzipped folder:
      Get-ChildItem -Path "PATH\TO\RAG Assistant" -Recurse | Unblock-File )
2. Unzip the folder anywhere.
3. Double-click "RAG Assistant.exe".
   If Windows says "Windows protected your PC", click "More info"
   then "Run anyway". This appears because the app isn't digitally signed.

Your documents are stored in:  C:\Users\<your name>\RAG Assistant

=====================================================================
RUNNING ON MAC (Apple Silicon: M1, M2, M3, M4 or newer)
=====================================================================
1. Unzip the file (double-click it) to get "RAG Assistant.app".
2. Move "RAG Assistant.app" to your Applications folder (optional).
3. Double-click it. macOS will block it the first time because it's
   from an unidentified developer. To allow it, use EITHER option:

   Option A (no typing):
      Open System Settings > Privacy & Security, scroll down, and
      click "Open Anyway" next to the message about RAG Assistant.
      Then open the app again and confirm.

   Option B (Terminal):
      Open Terminal, type the following, then a space:
         xattr -dr com.apple.quarantine
      Drag "RAG Assistant.app" into the Terminal window (this fills in
      its location), then press Return. Now open the app normally.

Your documents are stored in:  /Users/<your name>/RAG Assistant

=====================================================================
USING THE APP
=====================================================================
- The first launch needs an internet connection and may take a few
  minutes while it downloads two small AI models (about 180 MB).
  After that it works offline.
- Click "Add documents" to add PDFs or .txt files, then ask questions.
- Click "New conversation" to start a fresh topic.

=====================================================================
REQUIREMENTS
=====================================================================
- Windows 10/11, or a Mac with Apple Silicon (M1 or newer)
- At least 8 GB of RAM (16 GB recommended)
- AI models make the computer work hard. Keep laptops plugged in and
  on a hard surface so they can stay cool.