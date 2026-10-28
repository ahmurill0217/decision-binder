"""Build d1 prompt/completion datasets on CPU; never load model weights."""
from decision_binder.prompts import main

if __name__ == "__main__":
    main()
