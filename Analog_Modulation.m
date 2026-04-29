%% Analog Modulation and Demodulation Implementation
% ECE 342 Project - Spring 2026
% Team Project: Design and Implementation of Different Types of Analog 
% Modulation and Demodulation in Communication Systems Using AI
%
% This script implements: AM, FM, PM, DSB-SC, SSB, VSB
% Author: King Anwar
% Date: April 2026

clear all; close all; clc;

fprintf('=== PHASE 2: ANALOG MODULATION IMPLEMENTATION ===\n');
fprintf('ECE 342 Project - Spring 2026\n\n');

%% ========================================================================
% PARAMETERS AND SIGNAL SETUP
% ========================================================================
fs = 100e3;          % Sampling frequency (Hz) - at least 100 kHz
fc = 10e3;           % Carrier frequency (Hz) - 10 kHz
T = 0.1;             % Signal duration (seconds)
t = 0:1/fs:T-1/fs;   % Time vector (fixed to match length exactly)
N = length(t);       % Number of samples

% Message signal: sum of two sinusoids (representing audio)
f1 = 500;  % Message frequency 1 (Hz)
f2 = 1200; % Message frequency 2 (Hz)
m_t = 0.5*sin(2*pi*f1*t) + 0.3*sin(2*pi*f2*t);  % Message signal (normalized)

% Normalize message to [-1, 1] range for proper modulation
m_t = m_t / max(abs(m_t));

fprintf('Parameters: fs=%.0f Hz, fc=%.0f Hz, duration=%.3f s\n', fs, fc, T);
fprintf('Message frequencies: f1=%d Hz, f2=%d Hz\n', f1, f2);

%% ========================================================================
% HELPER FUNCTIONS
% ========================================================================

function [y_demod] = envelope_detector(y, fs, ~)
    % Simple envelope detector for AM
    % Rectification + low-pass filter
    y_rect = abs(y);  % Full-wave rectification
    % Design LPF for demodulation (cutoff ~ message bandwidth)
    fc_lpf = 2e3;  % 2 kHz cutoff (above max message freq)
    [b, a] = butter(5, fc_lpf/(fs/2), 'low');
    y_demod = filtfilt(b, a, y_rect);
end

function [y_demod] = coherent_demod(y, fs, fc, phase_offset)
    % Coherent/synchronous detector
    % Multiply by local carrier and LPF
    t = (0:length(y)-1)/fs;
    local_carrier = cos(2*pi*fc*t + phase_offset);
    y_product = y .* local_carrier;
    % LPF
    fc_lpf = 2e3;
    [b, a] = butter(5, fc_lpf/(fs/2), 'low');
    y_demod = 2 * filtfilt(b, a, y_product);
end

function [y_demod] = fm_discriminator(y, fs, fc)
    % Frequency discriminator for FM using gradient (more accurate)
    % 1. Convert to analytic signal using Hilbert transform
    y_analytic = hilbert(y);
    % 2. Extract instantaneous phase
    instantaneous_phase = unwrap(angle(y_analytic));
    % 3. Differentiate using gradient (rad/s)
    phase_deriv = gradient(instantaneous_phase, 1/fs);
    % 4. Convert to frequency deviation (Hz)
    freq_dev = phase_deriv / (2*pi);
    % 5. Remove carrier component (DC) and filter
    fc_lpf = 2e3;
    [b_hp, a_hp] = butter(5, fc_lpf/(fs/2), 'high');
    y_demod = filtfilt(b_hp, a_hp, freq_dev);
    [b_lp, a_lp] = butter(5, fc_lpf/(fs/2), 'low');
    y_demod = filtfilt(b_lp, a_lp, y_demod);
end

function [y_demod] = pm_demodulator(y, fs, fc)
    % Phase demodulation by extracting phase and removing carrier
    y_analytic = hilbert(y);
    instantaneous_phase = unwrap(angle(y_analytic));
    % Remove carrier phase component
    t = (0:length(y)-1)/fs;
    carrier_phase = 2*pi*fc*t;
    y_demod = instantaneous_phase - carrier_phase;
    % Remove DC and LPF
    fc_lpf = 2e3;
    [b_hp, a_hp] = butter(5, fc_lpf/(fs/2), 'high');
    y_demod = filtfilt(b_hp, a_hp, y_demod);
    [b_lp, a_lp] = butter(5, fc_lpf/(fs/2), 'low');
    y_demod = filtfilt(b_lp, a_lp, y_demod);
end

%% ========================================================================
% COMPUTE FFT FOR SPECTRA
% ========================================================================
N_fft = 2^nextpow2(N);
f_vec = (0:N_fft-1)*fs/N_fft - fs/2;
M_msg = fftshift(fft(m_t, N_fft));

%% ========================================================================
% MODULATION #1: AMPLITUDE MODULATION (AM)
% ========================================================================
fprintf('\n=== Modulation 1: AMPLITUDE MODULATION (AM) ===\n');

Ac = 1;                   % Carrier amplitude
modulation_index = 0.5;   % Modulation depth (0 to 1)

% Generate AM signal
s_AM = Ac * (1 + modulation_index*m_t) .* cos(2*pi*fc*t);

% Envelope detection (non-coherent)
s_AM_demod = envelope_detector(s_AM, fs, fc);

% For comparison: coherent demodulation
s_AM_coherent = coherent_demod(s_AM, fs, fc, 0);

% Compute FFT for spectra
M_AM = fftshift(fft(s_AM, N_fft));

figure(1); set(gcf, 'Name', 'AM Modulation');
subplot(3,2,1); plot(t*1000, m_t); title('Message Signal (Time)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,2); plot(f_vec/1000, abs(M_msg)); title('Message Signal (Spectrum)'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,3); plot(t*1000, s_AM); title('AM Modulated Signal (Time)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,4); plot(f_vec/1000, abs(M_AM)); title('AM Signal (Spectrum)'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,5); plot(t*1000, s_AM_demod); title('Demodulated Signal (Envelope)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,6); plot(t*1000, s_AM_coherent); title('Demodulated Signal (Coherent)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;

fprintf('AM: Carrier at %d Hz, Sidebands at %d Hz and %d Hz\n', fc, fc-f2, fc+f2);

%% ========================================================================
% MODULATION #2: FREQUENCY MODULATION (FM)
% ========================================================================
fprintf('\n=== Modulation 2: FREQUENCY MODULATION (FM) ===\n');

Ac = 1;
kf = 5e3;  % Frequency sensitivity (Hz/V) - peak deviation depends on m(t)

% Generate FM signal using direct frequency modulation
message_integral = cumsum(m_t) * (1/fs);  % Approximate integral
phi_FM = 2*pi*fc*t + 2*pi*kf*message_integral;
s_FM = Ac * cos(phi_FM);

% FM Discriminator demodulation
s_FM_demod = fm_discriminator(s_FM, fs, fc);

% Compute spectra
M_FM = fftshift(fft(s_FM, N_fft));
M_FM_demod = fftshift(fft(s_FM_demod, N_fft));

% Calculate Peak Frequency Deviation
delta_f = kf * max(m_t);
fprintf('FM: Peak frequency deviation = %.0f Hz\n', delta_f);
fprintf('FM: Carson bandwidth = %.0f Hz\n', 2*(delta_f + f2));

figure(2); set(gcf, 'Name', 'FM Modulation');
subplot(3,2,1); plot(t*1000, m_t); title('Message Signal (Time)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,2); plot(f_vec/1000, abs(M_msg)); title('Message Signal (Spectrum)'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,3); plot(t*1000, s_FM); title('FM Modulated Signal (Time)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,4); plot(f_vec/1000, abs(M_FM)); title('FM Signal (Spectrum)'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,5); plot(t*1000, s_FM_demod); title('Demodulated Signal (Discriminator)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,6); plot(f_vec/1000, abs(M_FM_demod)); title('Demodulated Signal (Spectrum)'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 10]);

%% ========================================================================
% MODULATION #3: PHASE MODULATION (PM)
% ========================================================================
fprintf('\n=== Modulation 3: PHASE MODULATION (PM) ===\n');

Ac = 1;
kp = pi;  % Phase sensitivity (radians/V)

% Generate PM signal
s_PM = Ac * cos(2*pi*fc*t + kp*m_t);

% PM Demodulation using phase extraction
s_PM_demod = pm_demodulator(s_PM, fs, fc);

% Compute spectra
M_PM = fftshift(fft(s_PM, N_fft));
M_PM_demod = fftshift(fft(s_PM_demod, N_fft));

figure(3); set(gcf, 'Name', 'PM Modulation');
subplot(3,2,1); plot(t*1000, m_t); title('Message Signal (Time)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,2); plot(f_vec/1000, abs(M_msg)); title('Message Signal (Spectrum)'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,3); plot(t*1000, s_PM); title('PM Modulated Signal (Time)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,4); plot(f_vec/1000, abs(M_PM)); title('PM Signal (Spectrum)'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,5); plot(t*1000, s_PM_demod); title('Demodulated Signal (Phase Extraction)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,6); plot(f_vec/1000, abs(M_PM_demod)); title('Demodulated Signal (Spectrum)'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 10]);

%% ========================================================================
% MODULATION #4: DSB SUPPRESSED CARRIER (DSB-SC)
% ========================================================================
fprintf('\n=== Modulation 4: DSB-SC ===\n');

Ac = 1;

% Generate DSB-SC signal (multiplies message by carrier, no carrier term)
s_DSB = Ac * m_t .* cos(2*pi*fc*t);

% DSB-SC requires COHERENT demodulation (synchronous detection)
s_DSB_demod = coherent_demod(s_DSB, fs, fc, 0);

% Try with phase error to show importance of coherent detection
s_DSB_demod_pe = coherent_demod(s_DSB, fs, fc, pi/4);  % 45 degree phase error

% Compute spectra
M_DSB = fftshift(fft(s_DSB, N_fft));
M_DSB_demod = fftshift(fft(s_DSB_demod, N_fft));

figure(4); set(gcf, 'Name', 'DSB-SC Modulation');
subplot(3,2,1); plot(t*1000, m_t); title('Message Signal'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,2); plot(f_vec/1000, abs(M_msg)); title('Message Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,3); plot(t*1000, s_DSB); title('DSB-SC Signal'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,4); plot(f_vec/1000, abs(M_DSB)); title('DSB-SC Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,5); plot(t*1000, s_DSB_demod); title('Demodulated (Phase Synced)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,6); plot(t*1000, s_DSB_demod_pe); title('Demodulated (45° Phase Error)'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;

%% ========================================================================
% MODULATION #5: SINGLE SIDEBAND (SSB) - USB AND LSB
% ========================================================================
fprintf('\n=== Modulation 5: SINGLE SIDEBAND (SSB) ===\n');

% Generate Hilbert transform for SSB
m_hilbert = hilbert(m_t);  % Analytic signal
m_ht = imag(m_hilbert);    % Hilbert transform (90 deg phase shift)

% CORRECTED: Upper Sideband (USB): m(t)cos(wc) - m_hat(t)sin(wc)
s_USB = 0.5 * m_t .* cos(2*pi*fc*t) - 0.5 * m_ht .* sin(2*pi*fc*t);
% Lower Sideband (LSB): m(t)cos(wc) + m_hat(t)sin(wc)
s_LSB = 0.5 * m_t .* cos(2*pi*fc*t) + 0.5 * m_ht .* sin(2*pi*fc*t);

% SSB Demodulation (coherent required)
s_USB_demod = coherent_demod(s_USB, fs, fc, 0);
s_LSB_demod = coherent_demod(s_LSB, fs, fc, 0);

% Compute spectra
M_USB = fftshift(fft(s_USB, N_fft));
M_LSB = fftshift(fft(s_LSB, N_fft));

% Plot USB
figure(5); set(gcf, 'Name', 'SSB-USB Modulation');
subplot(3,2,1); plot(t*1000, m_t); title('Message Signal'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,2); plot(f_vec/1000, abs(M_msg)); title('Message Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,3); plot(t*1000, s_USB); title('USB Signal'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,4); plot(f_vec/1000, abs(M_USB)); title('USB Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,5); plot(t*1000, s_USB_demod); title('Demodulated USB'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,6); plot(f_vec/1000, abs(fftshift(fft(s_USB_demod, N_fft)))); title('Demod Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 10]);

% Plot LSB
figure(6); set(gcf, 'Name', 'SSB-LSB Modulation');
subplot(3,2,1); plot(t*1000, m_t); title('Message Signal'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,2); plot(f_vec/1000, abs(M_msg)); title('Message Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,3); plot(t*1000, s_LSB); title('LSB Signal'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,4); plot(f_vec/1000, abs(M_LSB)); title('LSB Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,5); plot(t*1000, s_LSB_demod); title('Demodulated LSB'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,6); plot(f_vec/1000, abs(fftshift(fft(s_LSB_demod, N_fft)))); title('Demod Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 10]);

fprintf('SSB: Bandwidth = %.0f Hz (half of DSB)\n', f2);

%% ========================================================================
% MODULATION #6: VESTIGIAL SIDEBAND (VSB)
% ========================================================================
fprintf('\n=== Modulation 6: VESTIGIAL SIDEBAND (VSB) ===\n');

% Generate DSB signal (full both sidebands)
s_DSB_vsb = m_t .* cos(2*pi*fc*t);

% Design an asymmetric bandpass filter to create VSB:
% - Keep the upper sideband fully (fc to fc+1200 Hz)
% - Keep a vestige of the lower sideband (fc-500 to fc)
% - Use an FIR filter (firpm) for precise asymmetry
f_pass_low = fc - 500;   % lower passband edge (vestige)
f_stop_low = fc - 800;   % lower stopband edge (suppress deep LSB)
f_pass_high = fc + 1200; % upper passband edge (full USB)
f_stop_high = fc + 1500; % upper stopband edge

% Normalize frequencies to Nyquist (fs/2)
nyq = fs/2;
edges = [0, f_stop_low, f_pass_low, f_pass_high, f_stop_high, nyq] / nyq;
magnitudes = [0, 0, 1, 1, 0, 0];  % stop, transition, pass, pass, transition, stop

% Design FIR filter (default weight = 1 per band)
b_vsb = firpm(200, edges, magnitudes);   % <-- fixed: removed 'dev' argument
s_VSB = filter(b_vsb, 1, s_DSB_vsb);

% VSB Demodulation (coherent)
s_VSB_demod = coherent_demod(s_VSB, fs, fc, 0);

% Compute spectra
M_VSB = fftshift(fft(s_VSB, N_fft));
M_VSB_demod = fftshift(fft(s_VSB_demod, N_fft));

% Plot results
figure(7); set(gcf, 'Name', 'VSB Modulation');
subplot(3,2,1); plot(t*1000, m_t); title('Message Signal'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,2); plot(f_vec/1000, abs(M_msg)); title('Message Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,3); plot(t*1000, s_VSB); title('VSB Signal'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,4); plot(f_vec/1000, abs(M_VSB)); title('VSB Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 20]);
subplot(3,2,5); plot(t*1000, s_VSB_demod); title('Demodulated VSB'); xlabel('Time (ms)'); ylabel('Amplitude'); grid on;
subplot(3,2,6); plot(f_vec/1000, abs(M_VSB_demod)); title('Demod Spectrum'); xlabel('Frequency (kHz)'); ylabel('Magnitude'); grid on; xlim([0 10]);
%% ========================================================================
% SUMMARY COMPARISON
% ========================================================================
fprintf('\n=== MODULATION COMPARISON SUMMARY ===\n');
fprintf('Modulation Type | Bandwidth | Demod Method | Complexity\n');
fprintf('-------------|----------|-------------|------------\n');
fprintf('AM           | 2f_m (%d Hz) | Envelope    | Low\n', 2*f2);
fprintf('FM           | 2(D+ f_m)  | Discrim/PLL | Medium\n');
fprintf('PM           | Similar FM| PLL         | Medium\n');
fprintf('DSB-SC       | 2f_m (%d Hz) | Coherent   | High\n', 2*f2);
fprintf('SSB          | f_m (%d Hz)  | Coherent   | High\n', f2);
fprintf('VSB          | ~1.25f_m   | Coherent   | High\n');

fprintf('\n=== PHASE 2 COMPLETE ===\n');
fprintf('All modulation types simulated successfully.\n');
fprintf('Figures 1-7 show time-domain and frequency-domain results.\n');