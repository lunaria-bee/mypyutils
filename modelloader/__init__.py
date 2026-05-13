r'''Tooling to cache and stage HuggingFace models in background threads so
they're ready when you need them.

----------------------------------------------------------------
Introduction
----------------------------------------------------------------

``modelloader`` is intended to make up for potential deficiencies in compute
cluster storage. Generally, you want your HuggingFace cache location to be large
(because models are often large), persistent (so that your cache remains
available in future sessions), and fast (so that models load quickly). However,
you may not have access to storage that satisfies all three
properties. ``modelloader`` makes it easy to cache models in one location (the
*"cache directory"*) while loading them from another (the *"stage directory"*).

.. topic:: Desired directory properties:

    +---------------------+-------+------------+------+
    |                     | large | persistent | fast |
    +=====================+=======+============+======+
    | **cache directory** | ✅    | ✅         |      |
    +---------------------+-------+------------+------+
    | **stage directory** | ✅    |            | ✅   |
    +---------------------+-------+------------+------+

For example, on `Alliance Canada systems`_, Slurm jobs have access to large,
persistent, network-mounted storage via ``/scratch``, and large, volatile, local
storage via ``$SLURM_TMPDIR``. Thus, one could keep their HuggingFace cache
under ``/scratch``, and use ``modelloader`` to stage models to ``$SLURM_TMPDIR``
for loading at runtime.

.. _Alliance Canada systems: https://docs.alliancecan.ca/wiki/Using_node-local_storage

.. _modelloader-usage:

----------------------------------------------------------------
Usage
----------------------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModelLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main interface to ``modelloader`` is :class:`ModelLoader`.

.. code-block:: python

    from modelloader import ModelLoader

    # You must supply the stage directory (otherwise why are you using
    # modelloader?).
    loader = ModelLoader('/mnt/hf_stage')

    # You may optionally supply a cache directory to use other than the default
    # HuggingFace cache.
    alt_loader = ModelLoader(
        stage_dir='/mnt/hf_stage_2',
        cache_dir='/run/media/user/NAS/hf_cache',
    )

.. danger::

    Using the same cache and/or stage directories between two different
    instances of :class:`ModelLoader` is not supported, and the effects of doing
    so are undefined: It honestly could work fine, but I make no guarantees.

    .. code-block:: python

        # (Probably) don't do this.
        loader1 = ModelLoader('/mnt/hf_stage')
        loader2 = ModelLoader('/mnt/hf_stage', '/run/media/user/NAS/hf_cache')

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModelKey
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Models are identified using the :class:`ModelKey` type, which specifies a
HuggingFace Hub path and optionally a revision name, exactly the same as the
``pretrained_model_name_or_path`` and ``revision`` arguments to HuggingFace
:meth:`.from_pretrained() <transformers.PreTrainedModel.from_pretrained>`
methods. ``modelloader`` functions also accept :type:`KeyLike` values:

* a :class:`str` of the HuggingFace Hub path.
* a :class:`tuple` of the HuggingFace Hub path and the revision.

.. code-block:: python

    from modelloader import ModelKey

    # The following 4 lines are all equivalent:
    loader.stage(ModelKey('EleutherAI/pythia-160m'))
    loader.stage(ModelKey('EleutherAI/pythia-160m', None))
    loader.stage('EleutherAI/pythia-160m')
    loader.stage(('EleutherAI/pythia-160m', None))

    # As are the following 2 lines:
    loader.stage(ModelKey('EleutherAI/pythia-160m', 'step16'))
    loader.stage(('EleutherAI/pythia-160m', 'step16'))

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cache, stage, and load
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The three basic ``modelloader`` operations are cache, stage, and load:

* Use :meth:`.cache() <ModelLoader.cache>` to store models to the cache
  directory.
* Use :meth:`.stage() <ModelLoader.stage>` to copy models from the cache to the
  stage.
* Use :meth:`.load() <ModelLoader.load>` to load models from the stage into
  memory.

.. code-block:: python

    loader.cache('EleutherAI/pythia-160m')
    loader.stage('EleutherAI/pythia-160m')
    model, tokenizer = loader.load('EleutherAI/pythia-160m')

Operations always occur with the precedence cache → stage → load. Calling
:meth:`.stage() <ModelLoader.stage>` on an uncached model will automatically
cache it. Calling :meth:`.load() <ModelLoader.load>` on an unstaged model will
automatically stage it, also caching it if needbe.

.. code-block:: python

    # Directly loading a model checkpoint without caching or staging it first.
    model, tokenizer = loader.load(ModelKey('EleutherAI/pythia-160m', 'step2'))

.. admonition:: Pedantic details for silly nerds
    :class: dropdown

    The choice of the word "precedence" instead of "order" reflects some
    subtleties in how ``modelloader`` handles staging. When staging a model,
    ``modelloader`` checks the cache for missing/dirty files, copying good files
    from the cache directory to the stage directory, while simultaneously
    downloading any missing/dirty files directly to the stage directory, then
    copying them to the cache directory. Thus, a request to stage a model
    guarantees that it will also be cached, but the actual stage and cache
    operations are not handled distinctly from each other, and there is no
    guarantee as to the order in which the two halves of the combined
    stage+cache operation will complete.

    To get even more annoying about it: I wrote "cache → stage → load" up above
    because viewing ``modelloader`` as having a distinct order of operations
    like that is, in my opinion, the most straightforward way to look at things
    from a user perspective, but it's actually far more accurate to think of
    ``modelloader`` as having an implicational hierarchy of load(x) → stage(x) →
    cache(x): A request to stage a model is also a request to cache it, and a
    request to load a model is also a request to stage it.

    See the internal documentation in ``modelloader/threads.py`` for more.

:meth:`.cache() <ModelLoader.cache>` and :meth:`.stage() <ModelLoader.stage>`
both accept arbitrary numbers of keys. Operations occur in the same order as
they are given in method calls. Repeat calls to :meth:`.cache()
<ModelLoader.cache>` and :meth:`.stage() <ModelLoader.stage>` for the same
models are completely safe and do not cause repeated operations.

.. code-block:: python

    # The following will stage pythia-160m steps 1, 4, 16, 2, and 8, in that
    # order.
    loader.stage(
        ('EleutherAI/pythia-160m', 'step1'),
        ('EleutherAI/pythia-160m', 'step4'),
        ('EleutherAI/pythia-160m', 'step16'),
    )
    loader.stage(
        ('EleutherAI/pythia-160m', 'step2'),
        ('EleutherAI/pythia-160m', 'step8'),
    )

    # This does nothing (or more accurately, creates a stage operation request
    # that is simply ignored).
    loader.stage(('EleutherAI/pythia-160m', 'step1'))

:meth:`.cache() <ModelLoader.cache>` and :meth:`.stage() <ModelLoader.stage>`
return immediately, allowing ``modelloader`` to execute the requested operations
in the background. :meth:`.load() <ModelLoader.load>` blocks until the requested
model has been staged and loaded.

.. code-block:: python

    # Tell loader to stage desired models.
    loader.stage(
        ('EleutherAI/pythia-160m', 'step10000'),
        ('EleutherAI/pythia-160m', 'step11000'),
        ('EleutherAI/pythia-160m', 'step12000'),
    )

    # Load the first model and use it for computation.
    model, tokenizer = loader.load(('EleutherAI/pythia-160m', 'step10000'))
    some_long_operation(model, tokenizer)

    # While executing some_long_operation, loader has continued staging steps
    # 11000 and 12000 in the background, so this and subsequent .load() calls
    # should block for less time.
    model, tokenizer = loader.load(('EleutherAI/pythia-160m', 'step11000'))
    some_long_operation(model, tokenizer)

----------------------------------------------------------------
Threading
----------------------------------------------------------------

Each :meth:`ModelLoader` object maintains its own threads for carrying out
``modelloader`` operations; threads are automatically started when the object is
created, and automatically shut down when the object is destroyed [TODO make
this true]. As a user, you should never have to care about what these threads
are or what they're doing. If you intend to do ``modelloader`` development work,
or are just idly curiouis about ``modelloader``\ 's thread model, check the
internal documentation in ``modelloader/threads.py``.

'''

from .modelkey import ModelKey, KeyLike
from .modelloader import ModelLoader

__all__ = (
    'ModelLoader',
    'ModelKey',
    'KeyLike',
)
