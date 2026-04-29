
# Analog Modulation and Demodulation Using AI

This repository contains the complete implementation, analysis, and AI-enhanced demodulation of six classic analog modulation techniques. It is part of the **ECE 342 Project – Spring 2026** by Mohamed Anwar under the supervision of Dr. Ashraf Samy.

## Overview

The project explores:
- **Amplitude Modulation (AM)**
- **Frequency Modulation (FM)**
- **Phase Modulation (PM)**
- **Double Sideband Suppressed Carrier (DSB-SC)**
- **Single Sideband (SSB – USB/LSB)**
- **Vestigial Sideband (VSB)**

For each scheme, traditional demodulation methods are implemented. Additionally, a feed‑forward neural network (AI‑based demodulator) is trained to recover the original message from noisy modulated signals, offering **3–5 dB improvement** in low‑SNR regimes (0–10 dB).

## Repository Structure

```.
├── Analog_Modulation.m          # All modulation implementations
├── AI_Demodulation.m            # Neural network training and testing
├── Performance_Analysis.m       # Monte Carlo simulation & MSE vs SNR plots
├── report.docx                  # Full project report
├── README.md                    # This file
└── figures/                     # Generated time‑domain and frequency‑domain plots
```

##  Dependencies

- **MATLAB** (R2020b or later recommended)
- **Deep Learning Toolbox** (for neural network functions)
- **Signal Processing Toolbox** (for filtering, Hilbert transform, etc.)

## How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/muhamedanwer/Analog-Modulation-and-Demodulation-Using-AI.git
   cd Analog-Modulation-and-Demodulation-Using-AI
   ```

2. **Modulation & Demodulation**
   Run `Analog_Modulation.m` to generate all six modulated signals, perform traditional demodulation, and display time‑domain and frequency‑domain plots.

3. **AI‑Based Demodulation**
   Run `AI_Demodulation.m` to:
   - Train the neural network on noisy signals (SNR 0–20 dB)
   - Evaluate reconstruction quality
   - Generate training loss curves and example reconstructions

4. **Performance Comparison**
   Run `Performance_Analysis.m` to obtain the **MSE vs SNR** curves comparing traditional and AI‑enhanced demodulation across all modulation types.

## AI Model Architecture

- **Input**: 200‑sample downsampled modulated signal
- **Architecture**: 3‑layer feed‑forward network (`200 → 128 → 128 → 200`)
- **Training data**: 5000 samples per modulation type, mixed SNR between 0 and 20 dB
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam, learning rate 0.001

## Key Results

| Modulation | Bandwidth | Traditional MSE (0 dB) | AI MSE (0 dB) | Improvement |
|------------|-----------|------------------------|---------------|-------------|
| AM         | 2 fm      | ~0.15                  | ~0.08         | 3–5 dB      |
| FM         | see report| similar trend          | similar trend | 3–5 dB      |
| PM         | see report| similar trend          | similar trend | 3–5 dB      |
| DSB‑SC     | 2 fm      | similar trend          | similar trend | 3–5 dB      |
| SSB        | fm        | similar trend          | similar trend | 3–5 dB      |
| VSB        | ~1.25 fm  | similar trend          | similar trend | 3–5 dB      |

> The AI demodulator consistently outperforms traditional methods at low SNR, while both converge at high SNR (>15 dB).

##  Report

The full project report is available as `report.docx` and includes:
- Mathematical models of all six modulation schemes
- Detailed methodology and implementation notes
- Time‑domain waveforms and frequency spectra
- AI training curves and performance comparisons
- Discussion of results and limitations

##  Author

**Mohamed Anwar**  
ECE 342 Project – Spring 2026  
Supervised by Dr. Ashraf Samy

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Proakis, J.G. & Salehi, M. – *Digital Communications* (5th Ed.)
2. Carlson, A.B. – *Communication Systems* (McGraw-Hill)
3. Haykin, S. – *Communication Systems* (Wiley)
4. Lathi, B.P. – *Modern Digital and Analog Communication Systems* (Oxford)
5. Oppenheim, A.V. & Schafer, R.W. – *Discrete-Time Signal Processing*

---

For questions or collaboration, feel free to open an issue or reach out via the repository.
```
