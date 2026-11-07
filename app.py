"""Keep this process running: closing the browser does not unload the model."""
import multiprocessing
import threading
import webbrowser

if __name__ == '__main__':
    multiprocessing.freeze_support()
    import uvicorn
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:7860')).start()
    uvicorn.run('tts_lab.server:app', host='127.0.0.1', port=7860, workers=1, reload=False)
