# Continuous winner's curse: randomized winner selection with conditional CIs

`continuous_winners_curse_exponential_mechanism.ipynb` combines the earlier simulation and SPY real-data
notebooks and adds:

* **Selection.** The winning threshold is *sampled* from the exponential mechanism, a Gibbs kernel on the
  continuous domain with the adaptive scale from `src/Randomized_PSI.py`, discretized by quadrature.
  It replaces the argmax.
* **Inference.** A conditionally valid CI (and a median-unbiased estimate) for the true alpha
  θ(s_J) of the selected threshold. The CI comes from inverting the exact conditional pivot given J and the
  residual field.
* **Checks.** Coverage, regret and length versus ε. Coverage by selected location, and the grid-refinement
  check, on the simulation. A bootstrap-world calibration check on SPY.

Run from this folder (needs network access for `yfinance`; the CSVs it writes are git-ignored):

```bash
pip install numpy scipy pandas matplotlib yfinance jupyter
jupyter execute --inplace continuous_winners_curse_exponential_mechanism.ipynb
```
