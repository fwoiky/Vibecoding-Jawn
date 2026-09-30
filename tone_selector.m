% tone_selector.m
% A beginner-friendly script that asks the user for a number and plays
% a low tone (300 Hz) or a high tone (800 Hz) depending on the value.

clear;  % Remove any old variables from the workspace
clc;    % Clear the Command Window

%% Step 1: Ask the user for a numeric value
userValue = input('Enter a numeric value: ');

% Make sure the user actually typed a single number
if isempty(userValue) || ~isnumeric(userValue) || ~isscalar(userValue)
    error('Please run the script again and enter a single number.');
end

%% Step 2: Set up the sound settings
fs = 44100;         % Sampling frequency in Hz (samples per second)
duration = 1;       % Length of each tone in seconds
lowFreq = 300;      % Frequency of the low tone in Hz
highFreq = 800;     % Frequency of the high tone in Hz

% Create a time vector from 0 to 'duration' seconds, one point per sample
t = 0 : 1/fs : duration;

%% Step 3: Generate the two sine-wave tones
% A sine wave follows the formula: sin(2 * pi * frequency * time)
lowTone = sin(2 * pi * lowFreq * t);
highTone = sin(2 * pi * highFreq * t);

%% Step 4: Choose which tone to play based on the user's value
if userValue <= 20
    selectedTone = lowTone;
    condition = 'Value is less than or equal to 20 -> playing LOW tone (300 Hz)';
else
    selectedTone = highTone;
    condition = 'Value is greater than 20 -> playing HIGH tone (800 Hz)';
end

%% Step 5: Show the results in the Command Window
fprintf('You entered: %g\n', userValue);
fprintf('%s\n', condition);

%% Step 6: Play the selected tone through the speakers
sound(selectedTone, fs);
