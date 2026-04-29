%% AI-Based Demodulation Enhancement
% ECE 342 Project - Spring 2026
% AI Enhancement for AM Demodulation under Noisy Conditions
%
% This script implements a neural network for improved demodulation
% (No Communications Toolbox required; works with or without Deep Learning Toolbox)

clear all; close all; clc;

fprintf('=== AI-BASED DEMODULATION ENHANCEMENT ===\n');
fprintf('ECE 342 Project - Spring 2026\n\n');

%% ========================================================================
% Helper: Add AWGN without Communications Toolbox
% ========================================================================
function y_noisy = add_awgn(y, snr_db, ~)
    signal_power = mean(y.^2);
    noise_power = signal_power / (10^(snr_db/10));
    noise = sqrt(noise_power) * randn(size(y));
    y_noisy = y + noise;
end

%% ========================================================================
% PARAMETERS
% ========================================================================
fs = 100e3;       % Sampling frequency
fc = 10e3;        % Carrier frequency
T = 0.1;          % Duration
t = 0:1/fs:T-1/fs; % Time vector
N = length(t);

% Message parameters
frequencies = [300, 500, 800, 1200, 1500];
snr_range = 0:2:20;  % dB
num_trials = 50;

fprintf('AI Demodulation Enhancement for AM\n');
fprintf('====================================\n');

%% ========================================================================
% DATA GENERATION
% ========================================================================
fprintf('Generating training dataset...\n');

num_train = 5000;
num_test = 1000;
sequence_length = 200;

% ---------- FIX: Safe downsample factor ----------
if sequence_length > N
    error('sequence_length (%d) cannot exceed signal length N (%d). Increase T or reduce sequence_length.', sequence_length, N);
end
downsample_factor = floor(N / sequence_length);
if downsample_factor < 1
    downsample_factor = 1;
end
if downsample_factor * sequence_length > N
    downsample_factor = downsample_factor - 1;
end
% -------------------------------------------------
fprintf('Downsampling factor: %d -> output length %d\n', downsample_factor, sequence_length);

X_train = zeros(num_train, sequence_length);
y_train = zeros(num_train, sequence_length);
X_test  = zeros(num_test,  sequence_length);
y_test  = zeros(num_test,  sequence_length);

% Training data
for i = 1:num_train
    m = zeros(1, N);
    num_tones = randi([1,3]);
    tone_indices = randperm(length(frequencies), num_tones);
    for f = frequencies(tone_indices)
        m = m + rand()*sin(2*pi*f*t + rand()*2*pi);
    end
    m = m / max(abs(m));
    m_am = 0.5 + 0.5*m;
    
    s_AM = (1 + 0.5*m_am) .* cos(2*pi*fc*t);
    snr_db = snr_range(randi(length(snr_range)));
    s_noisy = add_awgn(s_AM, snr_db, 'measured');
    
    m_down = downsample(m, downsample_factor);
    s_down = downsample(s_noisy, downsample_factor);
    X_train(i, :) = s_down(1:sequence_length);
    y_train(i, :) = m_down(1:sequence_length);
end

% Test data
rng(42);
for i = 1:num_test
    m = zeros(1, N);
    num_tones = randi([1,3]);
    tone_indices = randperm(length(frequencies), num_tones);
    for f = frequencies(tone_indices)
        m = m + rand()*sin(2*pi*f*t + rand()*2*pi);
    end
    m = m / max(abs(m));
    m_am = 0.5 + 0.5*m;
    
    s_AM = (1 + 0.5*m_am) .* cos(2*pi*fc*t);
    snr_db = snr_range(randi(length(snr_range)));
    s_noisy = add_awgn(s_AM, snr_db, 'measured');
    
    m_down = downsample(m, downsample_factor);
    s_down = downsample(s_noisy, downsample_factor);
    X_test(i, :) = s_down(1:sequence_length);
    y_test(i, :) = m_down(1:sequence_length);
end

fprintf('Training samples: %d, Test samples: %d\n', num_train, num_test);
fprintf('Input sequence length: %d\n', sequence_length);

%% ========================================================================
% DEEP LEARNING MODEL
% ========================================================================
fprintf('\nBuilding Neural Network...\n');
loss_history = [];

if ~isempty(which('trainNetwork'))
    fprintf('Using MATLAB Deep Learning Toolbox (feedforward network)\n');
    
    layers = [
        featureInputLayer(sequence_length, 'Normalization', 'none')
        fullyConnectedLayer(256)
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(sequence_length)
        regressionLayer
    ];
    
    try
        net = trainNetwork(X_train, y_train, layers, ...
            trainingOptions('adam', ...
            'MaxEpochs', 30, ...
            'MiniBatchSize', 64, ...
            'InitialLearnRate', 1e-3, ...
            'Plots', 'training-progress', ...
            'Verbose', true));
        loss_history = 1:30; % placeholder
    catch ME
        fprintf('Training failed: %s\n', ME.message);
        fprintf('Falling back to custom network.\n');
        net = [];
    end
else
    fprintf('Deep Learning Toolbox not available. Using custom network.\n');
    net = [];
end

%% ========================================================================
% CUSTOM NEURAL NETWORK (FALLBACK)
% ========================================================================
if isempty(net)
    fprintf('\n=== TRAINING CUSTOM NETWORK ===\n');
    
    input_size = sequence_length;
    hidden_size = 128;
    output_size = sequence_length;
    
    W1 = randn(input_size, hidden_size) * sqrt(2/input_size);
    b1 = zeros(1, hidden_size);
    W2 = randn(hidden_size, hidden_size) * sqrt(2/hidden_size);
    b2 = zeros(1, hidden_size);
    W3 = randn(hidden_size, output_size) * sqrt(2/hidden_size);
    b3 = zeros(1, output_size);
    
    relu = @(x) max(0, x);
    relu_grad = @(x) double(x > 0);
    
    learning_rate = 0.001;
    epochs = 30;
    batch_size = 64;
    num_batches = floor(num_train / batch_size);
    
    % Normalize
    X_mean = mean(X_train(:));
    X_std = std(X_train(:));
    X_train_norm = (X_train - X_mean) / X_std;
    y_mean = mean(y_train(:));
    y_std = std(y_train(:));
    y_train_norm = (y_train - y_mean) / y_std;
    
    loss_history = [];
    for epoch = 1:epochs
        epoch_loss = 0;
        indices = randperm(num_train);
        for batch = 1:num_batches
            batch_idx = indices((batch-1)*batch_size + 1:batch*batch_size);
            X_batch = X_train_norm(batch_idx, :);
            y_batch = y_train_norm(batch_idx, :);
            
            % Forward
            z1 = X_batch * W1 + b1;
            a1 = relu(z1);
            z2 = a1 * W2 + b2;
            a2 = relu(z2);
            z3 = a2 * W3 + b3;
            y_pred = z3;
            
            loss = mean((y_pred - y_batch).^2, 'all');
            epoch_loss = epoch_loss + loss;
            
            % Backward
            delta3 = 2 * (y_pred - y_batch) / batch_size;
            dW3 = a2' * delta3;
            db3 = sum(delta3, 1);
            
            delta2 = (delta3 * W3') .* relu_grad(z2);
            dW2 = a1' * delta2;
            db2 = sum(delta2, 1);
            
            delta1 = (delta2 * W2') .* relu_grad(z1);
            dW1 = X_batch' * delta1;
            db1 = sum(delta1, 1);
            
            % Update
            W3 = W3 - learning_rate * dW3;
            b3 = b3 - learning_rate * db3;
            W2 = W2 - learning_rate * dW2;
            b2 = b2 - learning_rate * db2;
            W1 = W1 - learning_rate * dW1;
            b1 = b1 - learning_rate * db1;
        end
        avg_loss = epoch_loss / num_batches;
        loss_history = [loss_history, avg_loss];
        if mod(epoch,5)==0
            fprintf('Epoch %d/%d, Loss: %.4f\n', epoch, epochs, avg_loss);
        end
    end
    fprintf('Custom training complete!\n');
end

%% ========================================================================
% PERFORMANCE EVALUATION
% ========================================================================
fprintf('\n=== EVALUATING MODEL PERFORMANCE ===\n');

mse_traditional = zeros(length(snr_range), 1);
mse_ai = zeros(length(snr_range), 1);

for snr_idx = 1:length(snr_range)
    snr_db = snr_range(snr_idx);
    trial_mse_trad = 0;
    trial_mse_ai = 0;
    
    for trial = 1:num_trials
        % Generate test signal
        m = zeros(1, N);
        num_tones = randi([1,3]);
        tone_indices = randperm(length(frequencies), num_tones);
        for f = frequencies(tone_indices)
            m = m + rand()*sin(2*pi*f*t + rand()*2*pi);
        end
        m = m / max(abs(m));
        m_am = 0.5 + 0.5*m;
        
        s_AM = (1 + 0.5*m_am) .* cos(2*pi*fc*t);
        s_noisy = add_awgn(s_AM, snr_db, 'measured');
        
        % Traditional envelope detector
        s_demod_trad = abs(s_noisy);
        [b, a] = butter(5, 2e3/(fs/2), 'low');
        s_demod_trad = filtfilt(b, a, s_demod_trad);
        
        m_down = downsample(m, downsample_factor);
        trad_down = downsample(s_demod_trad, downsample_factor);
        m_down = m_down(1:sequence_length);
        trad_down = trad_down(1:sequence_length);
        trial_mse_trad = trial_mse_trad + mean((trad_down - m_down).^2);
        
        % AI Demodulation
        if isempty(net)  % custom network
            s_input = (s_noisy - X_mean) / X_std;
            s_down = downsample(s_input, downsample_factor);
            s_down = s_down(1:sequence_length);
            s_down = reshape(s_down, 1, sequence_length);
            z1 = relu(s_down * W1 + b1);
            z2 = relu(z1 * W2 + b2);
            ai_demod = (z2 * W3 + b3) * y_std + y_mean;
            trial_mse_ai = trial_mse_ai + mean((ai_demod - m_down).^2);
        else  % MATLAB network
            s_down = downsample(s_noisy, downsample_factor);
            s_down = s_down(1:sequence_length);
            ai_demod = predict(net, s_down);
            trial_mse_ai = trial_mse_ai + mean((ai_demod - m_down).^2);
        end
    end
    
    mse_traditional(snr_idx) = trial_mse_trad / num_trials;
    mse_ai(snr_idx) = trial_mse_ai / num_trials;
    fprintf('SNR: %d dB, Traditional MSE: %.4f, AI MSE: %.4f\n', ...
        snr_db, mse_traditional(snr_idx), mse_ai(snr_idx));
end

%% ========================================================================
% PLOT RESULTS
% ========================================================================
figure(8); set(gcf, 'Name', 'AI Demodulation Performance');
subplot(2,2,1);
plot(snr_range, mse_traditional, 'b-o', 'LineWidth', 2);
hold on;
plot(snr_range, mse_ai, 'r-s', 'LineWidth', 2);
xlabel('SNR (dB)');
ylabel('Mean Squared Error');
title('Demodulation MSE vs SNR');
legend('Traditional', 'AI-Enhanced');
grid on;

subplot(2,2,2);
semilogy(snr_range, mse_traditional, 'b-o', 'LineWidth', 2);
hold on;
semilogy(snr_range, mse_ai, 'r-s', 'LineWidth', 2);
xlabel('SNR (dB)');
ylabel('MSE (log scale)');
title('Demodulation MSE vs SNR (Log Scale)');
legend('Traditional', 'AI-Enhanced');
grid on;

% Example reconstruction at 10 dB SNR
subplot(2,2,3);
m_example = zeros(1, N);
for f = [500, 1200]
    m_example = m_example + sin(2*pi*f*t);
end
m_example = m_example / max(abs(m_example));
m_am_example = 0.5 + 0.5*m_example;
s_AM_example = (1 + 0.5*m_am_example) .* cos(2*pi*fc*t);
s_noisy_example = add_awgn(s_AM_example, 10, 'measured');

% Traditional
s_demod_trad = abs(s_noisy_example);
[b, a] = butter(5, 2e3/(fs/2), 'low');
s_demod_trad = filtfilt(b, a, s_demod_trad);
s_demod_trad = downsample(s_demod_trad, downsample_factor);
s_demod_trad = s_demod_trad(1:sequence_length);

% AI
if ~isempty(net)
    s_down = downsample(s_noisy_example, downsample_factor);
    s_down = s_down(1:sequence_length);
    s_demod_ai = predict(net, s_down);
else
    s_down = (downsample(s_noisy_example, downsample_factor) - X_mean) / X_std;
    s_down = s_down(1:sequence_length);
    s_down = reshape(s_down, 1, sequence_length);
    z1 = relu(s_down * W1 + b1);
    z2 = relu(z1 * W2 + b2);
    s_demod_ai = (z2 * W3 + b3) * y_std + y_mean;
end

t_ds = linspace(0, T, sequence_length);
plot(t_ds*1000, m_example(1:sequence_length), 'k-', 'LineWidth', 2);
hold on;
plot(t_ds*1000, s_demod_trad, 'b--', 'LineWidth', 1.5);
plot(t_ds*1000, s_demod_ai, 'r:', 'LineWidth', 1.5);
xlabel('Time (ms)');
ylabel('Amplitude');
title('Example Reconstruction at 10 dB SNR');
legend('Original', 'Traditional', 'AI');
grid on;

% Training loss
subplot(2,2,4);
if ~isempty(loss_history)
    plot(loss_history, 'b-', 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Loss (MSE)');
    title('Training Progress');
else
    text(0.5, 0.5, 'No training loss data', 'HorizontalAlignment', 'center');
    axis off;
end
grid on;

fprintf('\n=== COMPLETE ===\n');
fprintf('AI Demodulation training and evaluation complete.\n');
fprintf('Figure 8 shows MSE comparison and example reconstructions.\n');

%% ========================================================================
% SAVE TRAINED MODEL FOR PERFORMANCE EVALUATION
% ========================================================================
if ~isempty(net)
    % Deep Learning Toolbox model
    trained_model.net = net;
    trained_model.type = 'deep';
else
    % Custom network weights and normalization parameters
    trained_model.W1 = W1;  trained_model.b1 = b1;
    trained_model.W2 = W2;  trained_model.b2 = b2;
    trained_model.W3 = W3;  trained_model.b3 = b3;
    trained_model.X_mean = X_mean;  trained_model.X_std = X_std;
    trained_model.y_mean = y_mean;  trained_model.y_std = y_std;
    trained_model.type = 'custom';
end
save('ai_trained_model.mat', 'trained_model');
fprintf('Model saved as ai_trained_model.mat\n');