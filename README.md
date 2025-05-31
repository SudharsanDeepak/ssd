exp1
a

clc;
clear;
close all;

fs = 10000;
t = 0:1/fs:0.1;
fm = 50;
fc = 500;
A = 1;
m = 0.5;

message = cos(2 * pi * fm * t);
carrier = A * cos(2 * pi * fc * t);
am_signal = (1 + m * message) .* carrier;

demod_signal = abs(am_signal);

window_size = 100;  
demod_signal = filter(ones(1, window_size)/window_size, 1, demod_signal);
demod_signal = demod_signal - A;  

subplot(4,1,1);
plot(t, message);
title('Message Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4,1,2);
plot(t, carrier);
title('Carrier Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4,1,3);
plot(t, am_signal);
title('AM Modulated Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4,1,4);
plot(t, demod_signal);
title('Demodulated Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

exp 1
b

clc;
clear;
close all;

% Parameters
fs = 1000;                      % Sampling frequency
t = 0:1/fs:1;                   % Time vector

% Message Signal
am = 1;                         % Message amplitude
fm = 2;                         % Message frequency
msg = am * sin(2 * pi * fm * t);

% Carrier Signal 
ac = 2;                         % Carrier amplitude
fc = 20;                        % Carrier frequency
carrier = ac * sin(2 * pi * fc * t);

% Frequency Modulated Signal
beta = 5;                       % Modulation index
fmsig = ac * sin(2 * pi * fc * t + beta * sin(2 * pi * fm * t));

% Frequency Demodulated Signal (via envelope of differentiated signal)
diffsig = [0 diff(fmsig)];                             % Differentiate
rectified = abs(diffsig);                             % Rectify
window = 100;
demod = filter(ones(1, window)/window, 1, rectified); % Low-pass filter

% Plotting
subplot(4,1,1);
plot(t, msg);
title('Message Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4,1,2);
plot(t, carrier);
title('Carrier Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4,1,3);
plot(t, fmsig);
title('Frequency Modulated Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4,1,4);
plot(t, demod);
title('Frequency Demodulated Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

exp 2

clc; clear; close all;

% Sampling info
fs = 1000;               % Sampling frequency
t = 0:1/fs:1;            % Time vector

% Message signal
msg = sin(2*pi*2*t);     % 2 Hz sinusoidal message signal

% Parameters
pulse_interval = 20;     % Interval between pulses (samples)
pulse_width = 5;         % Width of pulse (samples)
threshold = 0.1;

% PAM (Pulse Amplitude Modulation)
pam = zeros(size(t));
pam(1:pulse_interval:end) = msg(1:pulse_interval:end);

% PWM (Pulse Width Modulation)
pwm = zeros(size(t));
for i = 1:pulse_interval:length(t)-pulse_width
    amp = msg(i);                  % Amplitude at that instant
    if amp > threshold
        width = round(pulse_width + abs(amp)*10);  % Vary width with amplitude
        pwm(i:i+width) = 1;
    end
end

% PPM (Pulse Position Modulation)
ppm = zeros(size(t));
for i = 1:pulse_interval:length(t)-pulse_width-10
    amp = msg(i);
    if amp > threshold
        delay = round(amp * 10);   % Vary position with amplitude
        ppm(i+10+delay:i+10+delay+pulse_width) = 1;
    end
end

% Plotting
subplot(4,1,1); plot(t,msg,'LineWidth',1.5); title('Message Signal'); grid on;
subplot(4,1,2); stem(t,pam,'Marker','none'); title('PAM Signal'); ylim([-1.5 1.5]); grid on;
subplot(4,1,3); stem(t,pwm,'Marker','none'); title('PWM Signal'); ylim([-0.2 1.2]); grid on;
subplot(4,1,4); stem(t,ppm,'Marker','none'); title('PPM Signal'); ylim([-0.2 1.2]); grid on;

exp 3

clc;
clear all;
close all;

% -------- Original Signal Parameters --------
Fs_original = 1000;              % Original (high) sampling frequency in Hz
t = 0:1/Fs_original:1;           % High-resolution time vector (1s duration)
f = 5;                           % Signal frequency in Hz
x = sin(2*pi*f*t);               % Continuous original signal

% -------- Sampling the Signal --------
Fs_sampled = 50;                 % Sampling frequency in Hz
ts = 0:1/Fs_sampled:1;           % Sampled time vector
xs = sin(2*pi*f*ts);             % Sampled signal values

% -------- Custom Sinc Function --------
mysinc = @(x) double(x == 0) + sin(pi*x)./(pi*x).*(x ~= 0);  % Handles x = 0 correctly

% -------- Sinc Interpolation for Reconstruction --------
xr = zeros(size(t));             % Initialize reconstructed signal
for n = 1:length(ts)
    xr = xr + xs(n) * mysinc(Fs_sampled * (t - ts(n)));
end

% -------- Plotting --------
figure;

subplot(3,1,1);
plot(t, x, 'b', 'LineWidth', 1.5);
title('Original Continuous-Time Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3,1,2);
stem(ts, xs, 'r', 'filled');
title('Sampled Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3,1,3);
plot(t, x, 'k--', 'LineWidth', 1); hold on;
plot(t, xr, 'g', 'LineWidth', 1.5);
legend('Original Signal', 'Reconstructed Signal');
title('Reconstructed Signal using Sinc Interpolation');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

exp4
a

clc;
clear all;
close all;

% -------- Message Signal --------
msgamplitude = 1;
msgfrequency = 2;
time = 0:0.001:1;
msgsignal = msgamplitude * sin(2 * pi * msgfrequency * time);

% -------- Sampling --------
samplingfrequency = 30;
samplinginterval = 1 / samplingfrequency;
samplestime = 0:samplinginterval:1;
samples = msgamplitude * sin(2 * pi * msgfrequency * samplestime);

% -------- Quantization --------
levels = 16;
quantised = round((samples + 1) * (levels / 2)) / (levels / 2) - 1;


% -------- Plotting --------
figure;

subplot(3,1,1);
plot(time, msgsignal, 'b', 'LineWidth', 1.5);
title('Original Message Signal');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(3,1,2);
stem(samplestime, samples, 'r', 'filled', 'LineWidth', 1.2);
title('Samples');
xlabel('Time (s)'); ylabel('Amplitude'); grid on; ylim([-1.5 1.5]);

subplot(3,1,3);
stem(samplestime, quantised, 'g', 'filled', 'LineWidth', 1.2);
title('Quantised');
xlabel('Time (s)'); ylabel('Amplitude'); grid on; ylim([-1.5 1.5]);

exp 4 
b

clc;
clear all;
close all;

% Message signal
A = 1;                % Amplitude
f = 2;                % Frequency (Hz)
t = 0:0.001:1;        % Time vector
msg = A * sin(2*pi*f*t);   % Original message signal

% Sampling
Fs = 40;                      % Sampling frequency
ts = 0:1/Fs:1;                % Sampling time vector
samples = A * sin(2*pi*f*ts); % Sampled signal

% Delta Modulation
delta = 0.1;                        % Step size
predict = zeros(1, length(samples));% Predicted signal
code = zeros(1, length(samples));   % Encoded bits (0 or 1)

for i = 2:length(samples)
    if samples(i) > predict(i-1)
        code(i) = 1;
        predict(i) = predict(i-1) + delta;
    else
        code(i) = 0;
        predict(i) = predict(i-1) - delta;
    end
end

% Plotting
figure;

subplot(3,1,1);
plot(t, msg, 'b', 'LineWidth', 1.5);
title('Original Message Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3,1,2);
stem(ts, samples, 'r', 'filled');
title('Sampled Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
ylim([-1.5 1.5]);

subplot(3,1,3);
stairs(ts, predict, 'g', 'LineWidth', 1.5);
title('Delta Modulated Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
ylim([-1.5 1.5]);

exp 5

clc;
clear;
close all;

% Binary message
data = [1 0 1 1 0 1 0 0 1 1];
bitrate = 1;
bit_duration = 1 / bitrate;

% Time and carrier parameters
fs = 1000;  % samples per second
t = 0:1/fs:bit_duration - 1/fs;

% Carrier frequencies
fc1 = 5; % frequency for bit 1
fc0 = 2; % frequency for bit 0
A1 = 1;  % amplitude for bit 1
A0 = 0;  % amplitude for bit 0

% Initialize modulated signals
ASK = [];
FSK = [];
PSK = [];
message = [];

for i = 1:length(data)
    if data(i) == 1
        ask_sig = A1 * sin(2 * pi * fc1 * t);
        fsk_sig = sin(2 * pi * fc1 * t);
        psk_sig = sin(2 * pi * fc1 * t);
        msg = ones(1, length(t));
    else
        ask_sig = A0 * sin(2 * pi * fc1 * t);
        fsk_sig = sin(2 * pi * fc0 * t);
        psk_sig = sin(2 * pi * fc1 * t + pi);
        msg = zeros(1, length(t));
    end
    ASK = [ASK ask_sig];
    FSK = [FSK fsk_sig];
    PSK = [PSK psk_sig];
    message = [message msg];
end

% Time vector
total_time = 0:1/fs:bit_duration*length(data) - 1/fs;

% Plot all graphs in vertically stacked form
figure;
subplot(4,1,2);
plot(total_time, FSK, 'LineWidth', 1);
title('FSK Modulation'); xlabel('Time'); ylabel('Amplitude'); grid on;

subplot(4,1,3);
plot(total_time, PSK, 'LineWidth', 1);
title('PSK Modulation'); xlabel('Time'); ylabel('Amplitude'); grid on;

subplot(4,1,4);
plot(total_time, ASK, 'LineWidth', 1);
title('ASK Modulation'); xlabel('Time'); ylabel('Amplitude'); grid on;

subplot(4,1,1);
plot(total_time, message, 'LineWidth', 2);
title('Message Bits'); xlabel('Time'); ylabel('Bits'); grid on;
axis([0 bit_duration*length(data) -0.5 1.5]);

exp 6 

clc; clear; close all;

 

data = randi([0 1], 1, 1000); % Random binary data

mod_type = input('Enter modulation type [1=BPSK, 2=QPSK, 3=16QAM, 4=64QAM]: ');

 

% Normalization and bits per symbol

norm_factors = [1.0, 0.7071, 0.3162, 0.1543];

bits_per_symbol = [1, 2, 4, 6];

 

k = norm_factors(mod_type);

mode = bits_per_symbol(mod_type);

 

% Define constellation

switch mode

    case 1  % BPSK

        symbols = k * [1, -1];

    case 2  % QPSK

        symbols = k * [1+1i, -1+1i, 1-1i, -1-1i];

    case 4  % 16-QAM

        re = [-3 -1 1 3]; im = [-3 -1 1 3];

        [I, Q] = meshgrid(re, im);

        symbols = k * (I(:) + 1i*Q(:));

    case 6  % 64-QAM

        re = [-7 -5 -3 -1 1 3 5 7]; im = re;

        [I, Q] = meshgrid(re, im);

        symbols = k * (I(:) + 1i*Q(:));

end

 

% Pad data if not divisible

extra = mod(length(data), mode);

if extra ~= 0

    data = [data, zeros(1, mode - extra)];

end

 

% Convert bits to decimal manually (no bi2de)

data = reshape(data, mode, []).';

decimal_vals = zeros(1, size(data,1));

for i = 1:mode

    decimal_vals = decimal_vals + data(:,i) * 2^(mode - i);

end

decimal_vals = decimal_vals + 1; % MATLAB indexing

 

% Map to symbols

map_out = symbols(decimal_vals);

 

% Plot

figure;

plot(real(map_out), imag(map_out), '.', 'MarkerSize', 10, 'Color', [0 0 0.7]);
xlabel('In-phase Amplitude'); ylabel('Quadrature Amplitude'); grid on;

exp 7 
a


% Linear block code: Decode Message without syndrome

clc;
clear;

k = input('Enter the length of message word: ');
n = input('Enter the length of codeword: ');
disp('Enter the parity matrix (as a k x (n-k) matrix): ');
p = input('');

% Generator matrix G = [I | P]
G = [eye(k) p];

m = input('Enter the message word (row vector of length k): ');

% Encoding: c = m * G mod 2
c = mod(m * G, 2);
disp('Encoded codeword:');
disp(c);

% Simulate received codeword (you can flip bits manually if needed)
r = c; % For now assume no error
disp('Received codeword (assumed same as encoded):');
disp(r);

% Decoding: take first k bits (systematic code)
d = r(1:k);
disp('Decoded message:');
disp(d);


b

% SIVASUBRAMANIAN V 23EC148
% Cyclic Code

close all;
clear all;
clc;

k = input('Enter the length of the message word: ');
n = input('Enter the length of the code word: ');
m = input('Enter the message word: ');

G = cyclpoly(n, k, 'max');
gx = poly2sym(G);
disp('Generator polynomial:');
disp(gx);

c = encode(m, n, k, 'cyclic', G);
disp('Encoded cyclic codeword:');
disp(c);

D = decode(c, n, k, 'cyclic', G);
disp('Decoded message:');
disp(D);

c

% SIVASUBRAMANIAN V 23EC148
% Linear block code: Decode Message without syndrome

close all;
clear all;
clc;

k = input('Enter the length of message word: ');
n = input('Enter the length of codeword: ');
p = input('Enter the parity matrix: ');
G = [eye(k) p];

m = input('Enter the msg word: ');
c = encode(m, n, k, 'linear', G);
disp('Encoded codeword:');
disp(c);

d = decode(c, n, k, 'linear', G);
disp('Decoded message:');
disp(d);

exp 8 
a

clc;
clear;
x = input('Enter the input sequence x[n] as a vector (e.g., [5 3 2 1]): ');
h = input('Enter the impulse response h[n] as a vector (e.g., [1 2 3 4]): ');
linear_conv = conv(x, h);
disp('Linear Convolution Result:');
disp(linear_conv);
fprintf('Length of Linear Convolution Output: %d\n', length(linear_conv));
figure;
subplot(3,1,1);
stem(0:length(x)-1, x, 'filled', 'Color', 'k', 'LineWidth', 1.5);
title('Input Signal x[n]');
xlabel('n');
ylabel('Amplitude');
grid on;
subplot(3,1,2);
stem(0:length(h)-1, h, 'filled', 'Color', 'k', 'LineWidth', 1.5);
title('Input Signal h[n]');
xlabel('n');
ylabel('Amplitude');
grid on;
subplot(3,1,3);
stem(0:length(linear_conv)-1, linear_conv, 'filled', 'Color', 'k', 'LineWidth', 1.5);
title('Linear Convolution Output y[n]');
xlabel('n');
ylabel('Amplitude');
grid on;

b

x = [1 2 3];
h = [4 5];
y_linear = conv(x, h);
disp('Linear Convolution:');
disp(y_linear);
x = [5 3 2 1];
h = [1 2 3 4];
N = max(length(x), length(h));  % or any desired length
y_circular = cconv(x, h, N);
disp('Circular Convolution:');
disp(y_circular);
figure;
subplot(3,1,1);
stem(0:N-1,x,"filled",'color','k','LineWidth',2);
grid on;
subplot(3,1,2);
stem(0:N-1,h,"filled",'color','k','LineWidth',2);
grid on;
subplot(3,1,3);
stem(0:N-1,y_circular,"filled",'color','k','LineWidth',2);
grid on;
