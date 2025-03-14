RAG Application
----------------

.. automodule:: classes.rag_application
    :members:
    :undoc-members:
    :private-members:
    :special-members: __init__
    :show-inheritance:

Doctests
========

.. testsetup::

    from classes.rag_application import RAGApplication

.. doctest::

    >>> retriever, rag_chain, web_documents = None, None, []
    >>> app = RAGApplication(retriever, rag_chain, web_documents)
    >>> isinstance(app, RAGApplication)
    True
