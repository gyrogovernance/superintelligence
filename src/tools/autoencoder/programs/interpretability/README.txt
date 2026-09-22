Interpretability tools (weight geometry)

These scripts inspect frozen Bonsai / autoencoder weights. They are not the genomics wet-lab.

  python -m src.tools.autoencoder.programs.interpretability.census
  python -m src.tools.autoencoder.programs.interpretability.weight_probe

Files
  adapter.py     load weight rows from GGUF
  converter.py   map rows onto CGM tiles / shells
  census.py      full weight survey (+ --combine for shard merge)
  weight_probe.py deeper probe of selected tensors

Old names merge_census and cooperative were removed. Do not reintroduce chat-jargon filenames.
