% basic_matlab_demo.m
% A beginner-friendly MATLAB script that shows how to:
%   1. Create variables
%   2. Do basic arithmetic
%   3. Build a numeric vector
%   4. Find its mean, minimum, and maximum
%   5. Print everything neatly with fprintf
%
% How to run: open this file in MATLAB and press "Run",
% or type  basic_matlab_demo  in the Command Window.

% ---- Start fresh ----
clc;        % clear the Command Window so the output is easy to read
clear;      % remove any old variables from the workspace

%% Step 1: Define two variables
% A variable is a name that stores a value. The semicolon (;) at the
% end of a line stops MATLAB from printing the value automatically.
a = 12;
b = 5;

fprintf('===== Step 1: Variables =====\n');
fprintf('a = %d\n', a);     % %d is a placeholder for a whole number
fprintf('b = %d\n\n', b);   % \n starts a new line

%% Step 2: Basic arithmetic
% MATLAB uses the usual math symbols: + - * / and ^ (power).
sumResult        = a + b;   % addition
differenceResult = a - b;   % subtraction
productResult    = a * b;   % multiplication
quotientResult   = a / b;   % division (gives a decimal: 2.4)
powerResult      = a ^ 2;   % a squared

fprintf('===== Step 2: Arithmetic =====\n');
fprintf('%d + %d = %d\n', a, b, sumResult);
fprintf('%d - %d = %d\n', a, b, differenceResult);
fprintf('%d * %d = %d\n', a, b, productResult);
fprintf('%d / %d = %.2f\n', a, b, quotientResult);  % %.2f = decimal, 2 places
fprintf('%d ^ 2 = %d\n\n', a, powerResult);

%% Step 3: Create a numeric vector
% A vector is a list of numbers inside square brackets [ ].
scores = [85, 92, 78, 64, 99, 71];

fprintf('===== Step 3: Vector =====\n');
fprintf('scores = ');
fprintf('%d ', scores);     % fprintf repeats the format for every element
fprintf('\n');
fprintf('Number of elements: %d\n\n', numel(scores));  % numel counts items

%% Step 4: Vector statistics
% MATLAB has built-in functions that work on a whole vector at once.
avgScore = mean(scores);    % the average value
minScore = min(scores);     % the smallest value
maxScore = max(scores);     % the largest value

fprintf('===== Step 4: Statistics =====\n');
fprintf('Mean    : %.2f\n', avgScore);
fprintf('Minimum : %d\n', minScore);
fprintf('Maximum : %d\n\n', maxScore);

fprintf('Done! Try changing a, b, or scores and run the script again.\n');
