function run_capsule(varargin)
% RUN_CAPSULE  Code Ocean entry point for running GIAnT:
%   correct motion with StripRegistration and then extract sources with
%   SILo.
%
%   Parameters arrive as name/value pairs from the `run` shell script.
%
%   Parameters are split into three groups:
%     stripReg*  -> StripRegistration only
%     silo*   -> SILo only
%     shared* -> both functions accept it, so it is copied into both
% 

%% ------------------------------------------------------------------
%  Parameter definitions
%  ------------------------------------------------------------------

% StripRegistration parameters. These carry explicit defaults and are
% ALWAYS forwarded, so StripRegistration is fully specified from here.
stripRegNumericDefaults = struct( ...
    'maxshift',    50, ...
    'clipShift',   10, ...
    'removeLines',  4, ...
    'ds_time',      1, ...
    'frameRate',    0);

stripRegLogicalDefaults = struct( ...
    'overwriteExisting', true, ...
    'saveTif',           false);

% Taken by both functions
sharedNumericDefaults = struct( ...
    'nWorkers', 1);

% SILo parameters. No defaults here on purpose: these default to [] and
% are only forwarded when the caller actually supplies a value, so SILo's
% own internal defaults stay in control for anything left unset.
siloNumericFields = { ...
    'sigma_px', ...
    'nmfIter', ...
    'dXY', ...
    'photonScale', ...
    'lambda', ...
    'phi', ...
    'denoiseWindow_s', ...
    'baselineWindow_Glu_s', ...
    'baselineWindow_Ca_s', ...
    'activityChannel', ...
    'tau_s', ...
    'tau2_s', ...
    'VIF', ...
    'peakth', ...
    'minPeakDistance', ...
    'motionThresh', ...
    'analyzeHz', ...
    'nanThresh', ...
    'discardInitial_s'};

siloLogicalFields = { ...
    'includeIntegrationROIs', ...
    'drawUserRois', ...
    'isSLAP2'};

%% ------------------------------------------------------------------
%  Parse inputs
%  ------------------------------------------------------------------
%  Validators deliberately accept char/string as well as numeric/logical:
%  values coming from the shell may arrive as text. Conversion to the real
%  type happens in toNumber/toLogical when the structs are assembled.

charValidator    = @(x) ischar(x) || isstring(x);
numericValidator = @(x) isnumeric(x) || ischar(x) || isstring(x);
logicalValidator = @(x) islogical(x) || isnumeric(x) || ischar(x) || isstring(x);

p = inputParser;
addParameter(p, 'inputDir',  '/data/',    charValidator);
addParameter(p, 'resultDir', '/results/', charValidator);

% StripRegistration and shared parameters register with their defaults
fields = fieldnames(stripRegNumericDefaults);
for i = 1:numel(fields)
    addParameter(p, fields{i}, stripRegNumericDefaults.(fields{i}), numericValidator);
end

fields = fieldnames(stripRegLogicalDefaults);
for i = 1:numel(fields)
    addParameter(p, fields{i}, stripRegLogicalDefaults.(fields{i}), logicalValidator);
end

fields = fieldnames(sharedNumericDefaults);
for i = 1:numel(fields)
    addParameter(p, fields{i}, sharedNumericDefaults.(fields{i}), numericValidator);
end

% SILo parameters register with [] so "not supplied" stays distinguishable
for i = 1:numel(siloNumericFields)
    addParameter(p, siloNumericFields{i}, [], numericValidator);
end

for i = 1:numel(siloLogicalFields)
    addParameter(p, siloLogicalFields{i}, [], logicalValidator);
end

parse(p, varargin{:});

%% ------------------------------------------------------------------
%  Assemble the two parameter structs
%  ------------------------------------------------------------------

% StripRegistration: every field is present, either the default or the
% caller's value.
stripRegParams = struct();

fields = fieldnames(stripRegNumericDefaults);
for i = 1:numel(fields)
    stripRegParams.(fields{i}) = toNumber(p.Results.(fields{i}), fields{i});
end

fields = fieldnames(stripRegLogicalDefaults);
for i = 1:numel(fields)
    stripRegParams.(fields{i}) = toLogical(p.Results.(fields{i}), fields{i});
end

% SILo: only fields the caller supplied are copied across; the rest are
% left absent so SILo falls back to its own defaults.
siloParams = struct();

for i = 1:numel(siloNumericFields)
    val = p.Results.(siloNumericFields{i});
    if ~isempty(val)
        siloParams.(siloNumericFields{i}) = toNumber(val, siloNumericFields{i});
    end
end

for i = 1:numel(siloLogicalFields)
    val = p.Results.(siloLogicalFields{i});
    if ~isempty(val)
        siloParams.(siloLogicalFields{i}) = toLogical(val, siloLogicalFields{i});
    end
end

% Capsule-level overrides: fixed properties of this pipeline rather than
% tunables. Set unconditionally AFTER the loops above, so they win over
% anything the caller passed for these four names.
%   isSLAP2                - this capsule does not process SLAP2 data
%   drawUserRois           - no annotations asset, so no user ROIs exist
%   includeIntegrationROIs - integration ROIs are only relevant for SLAP2
%   analyzeHz              - analysis rate can only distinct from imaging
%                            rate for SLAP2 data
siloParams.isSLAP2 = false;
siloParams.drawUserRois = false;
siloParams.includeIntegrationROIs = false;
siloParams.analyzeHz = stripRegParams.frameRate;

% Shared parameters go into both structs
fields = fieldnames(sharedNumericDefaults);
for i = 1:numel(fields)
    val = toNumber(p.Results.(fields{i}), fields{i});
    stripRegParams.(fields{i}) = val;
    siloParams.(fields{i})  = val;
end

% Echo the resolved parameters into the capsule log for provenance
disp('StripRegistration params:')
disp(stripRegParams)
disp('SILo params:')
disp(siloParams)

%% ------------------------------------------------------------------
%  Find session directories and process each one
%  ------------------------------------------------------------------
%  Every immediate subdirectory of inputDir is treated as one session.

mainDir = p.Results.inputDir;

contents = dir(fullfile(mainDir, '*'));
dirNames = {contents([contents.isdir]).name};
inputDirs = {};
for i = 1:length(dirNames)
    if strcmp(dirNames{i},'.') || strcmp(dirNames{i},'..')
        continue;
    end
    inputDirs{end+1} = fullfile(mainDir,dirNames{i}); %#ok<AGROW>
end

for ix = 1:length(inputDirs)
    processDirectory(inputDirs{ix}, p.Results.resultDir, stripRegParams, siloParams);
end

end

function processDirectory(inputDir, resultDir, stripRegParams, siloParams)
% PROCESSDIRECTORY  Run the full pipeline on a single session directory.
%
%   Results go to resultDir/<session name>/. StripRegistration and SILo
%   both operate in place on trial_table.h5, so SILo picks up whatever
%   StripRegistration wrote.

    disp(['Processing ' inputDir]);

    % fileparts returns an empty name when the path ends in a separator
    if endsWith(inputDir, filesep)
        inputDir = inputDir(1:end-1);
    end
    [~, dataName] = fileparts(inputDir);

    % Mirror the session name in the results tree
    outputDir = fullfile(resultDir, dataName);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Get all files in the folder (no subfolders)
    files = dir(fullfile(inputDir, '*'));

    % Filter for extension AND exclude 'groundtruth.h5'
    validFiles = files(~[files.isdir] & ...
        (endsWith({files.name}, {'.h5', '.tif', '.tiff'}, 'IgnoreCase', true)) & ...
        ~contains({files.name}, 'groundtruth'));

    % Only the first match is processed: a session is expected to hold
    % exactly one data file
    if ~isempty(validFiles)
        dataFile = validFiles(1).name;
        disp(dataFile);
    else
        disp(['No valid data files found in ' inputDir]);
        return;
    end

    % Build the trial table that both downstream steps read and write
    buildTrialTable(inputDir, outputDir, false, dataFile);

    StripRegistration(fullfile(outputDir,'trial_table.h5'), stripRegParams);

    SILo(fullfile(outputDir,'trial_table.h5'), siloParams);
end

function y = toNumber(x, fieldName)
% TONUMBER  Coerce a parameter value to numeric, naming the offending
%   field on failure. Needed because shell-supplied values may arrive as
%   char/string rather than as numbers.

    if isnumeric(x)
        y = x;
        return;
    end

    if ischar(x) || isstring(x)
        y = str2double(x);

        if isnan(y)
            error('Parameter "%s" must be numeric, but got "%s".', fieldName, string(x));
        end

        return;
    end

    error('Parameter "%s" must be numeric.', fieldName);
end

function y = toLogical(x, fieldName)
% TOLOGICAL  Coerce a parameter value to logical. Accepts true/false, 1/0,
%   and the strings "true"/"false"/"1"/"0"/"yes"/"no", so the same flag
%   works whether it comes from MATLAB or from the shell.

    if islogical(x)
        y = x;
        return;
    end

    if isnumeric(x)
        if x == 0
            y = false;
            return;
        elseif x == 1
            y = true;
            return;
        else
            error('Parameter "%s" must be logical-like: true, false, 1, or 0.', fieldName);
        end
    end

    if ischar(x) || isstring(x)
        s = lower(strtrim(string(x)));

        if s == "true" || s == "1" || s == "yes"
            y = true;
            return;
        elseif s == "false" || s == "0" || s == "no"
            y = false;
            return;
        else
            error('Parameter "%s" must be logical-like, but got "%s".', fieldName, string(x));
        end
    end

    error('Parameter "%s" must be logical-like.', fieldName);
end
