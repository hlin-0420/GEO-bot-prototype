Usage
=====

Installation
------------

To use the GEO chatbot, first clone the following repository from Git:

.. code-block:: console

   (.venv) $ git clone https://github.com/hlin-0420/GEO-bot-prototype.git

Change directory to the cloned project location

.. code-block:: console

   (.venv) $ cd GEO-bot-prototype

Set up a virtual environment to prepare for installing relevant libraries.

.. code-block:: console

   (.venv) $ python -m venv venv

Windows

.. code-block:: console

   (.venv) $ \venv\Scripts\activate 


macOS / Linux

.. code-block:: console

   (.venv) $ source /venv/bin.activate

Install all the relevant dependencies from requirements.txt file.

.. code-block:: console

   (.venv) $ pip install -r requirements.txt

This project uses **Llama 3.2**, **Deepseek 1.5**, and **OpenAI** models, install them through:

.. code-block:: console

   (.venv) $ ollama pull llama3.2:latest

   (.venv) $ ollama pull deepseek-r1:1.5b

OpenAI uses the OpenAI API with its keys configured through the following instructions:

.. code-block:: console

   (.venv) $ export OPENAI_API_KEY="your-api-key-here"  # macOS/Linux

   (.venv) $ set OPENAI_API_KEY="your-api-key-here"  # Windows

Run the application through 

.. code-block:: console

   (.venv) $ python offline-app.py