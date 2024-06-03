### Access Google Gemini models such as:
- gemini-pro
- gemini-pro-vision

### Environment setup
- Option 1: (have existing virtual env)
    - Use exisiting [pipfile/pipfile.lock file](https://stackoverflow.com/questions/52171593/how-to-install-dependencies-from-a-copied-pipfile-inside-a-virtual-environment)
- Option 2: (don't have existing virtual env)
    - Install [pipenv](https://pipenv.pypa.io/en/latest/).
        ```
        pip install pipenv
        ```
    - Install a package (creates env).
        ```
        python -m pipenv install google-generativeai
        ```
    - Activate pipenv environment.
        ```
        python -m pipenv shell
        ```
    - Install other packages.
        ```
        pipenv install google-generativeai python-dotenv
        ```

### Run streamlit app
- Provide API key (https://ai.google.dev/) in .env file.
- Start streamlit app.
    ```
    streamlit run gemini_simple_app.py
    ```