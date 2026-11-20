"""Models.

gbm.py   LightGBM baseline on window statistics (point model + quantile models)
nn.py    mean-variance wrapper + training loop shared by all deep backbones, `build_neural`
tcn.py   temporal convolutional network backbone (main model)
lstm.py  LSTM backbone (challenger)

To add a deep challenger: write a backbone module that maps (batch, window, features) to
(batch, out_dim), set `self.out_dim`, and register it in `nn.BACKBONES`. Ensembles,
MC dropout and conformal calibration then work for it without further changes.
"""
