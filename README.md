# heart-sound-analysis
Showcase of the central analytical tool used extensively throughout MEng project 'Understanding heart sound propagation in the human chest'.

The analtyical tools cover the following:

*Heart Sounds Testing*

- Algorithm is used to segment time-series heart sound data in to systole, diastole, S1 peak and S2 peaks.
- A SNR calculation regime is proposed to normalise heart sound volume analysis.
- Heart sound amplitude heatmaps are generated.

*Forced Vibrational Tests*

- Forcing input and accelerometer output time series data is processed in the frequency domain to uncover frequency dependent behaviour.
- Welch's method averging is used to filter out background noise power.
- Complex PSDs are found for the input forcing and acceleration responses.
- Frequency response functions and coherence plots are generated for the 1D vibrational experiments.
- Circle-fitting regression is used to estimate the frequency and damping of resonant peaks present in the FRFs.
